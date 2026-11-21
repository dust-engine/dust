//! The console's socket. Started with `DUST_CONSOLE_SOCKET` set to a path, the
//! game listens on a Unix domain socket there (Windows has them too, AF_UNIX,
//! since Windows 10 1803) and runs the command lines sent to it, as the
//! console's prompt runs those typed there, answering each with what it
//! printed. `dustctl` sends them.
//!
//! The protocol is plain text. A client sends console lines, each ended by a
//! newline, as they would be typed at the prompt. The console answers each, in
//! order, with a status line, `ok <length>` or `err <length>`, then that many
//! bytes of output and a newline:
//!
//! ```text
//! → echo hello world
//! ← ok 11
//! ← hello world
//! ```
//!
//! The length is the output's size in bytes, without the newline that ends
//! the answer, so the output can hold newlines of its own. A command that
//! failed answers `err`, with why it failed as its output.
//!
//! Between answers, the game also sends each message logged while the client
//! is connected, as it's logged, the same way with `log`: the message as text,
//! much as the game prints it to stderr. A line's log messages come before its
//! answer. A client that doesn't want them skips them; one that falls more
//! than [`MAX_ROWS`](super::MAX_ROWS) messages behind loses the oldest.
//!
//! ```text
//! → log -l warn hi
//! ← log 55
//! ← 0.699  WARN command{line="log -l warn hi"}: console: hi
//! ← ok 0
//! ←
//! ```
//!
//! Tasks on the UI thread serve the socket for as long as the app runs, whether
//! or not DevTools shows the console, waiting for it through async-io. Each
//! connection gets a task, which runs its lines one at a time with
//! [`command::run`], as the console's prompt does.

use std::io;
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_io::{Async, Timer};
use futures_lite::StreamExt as _;
use futures_lite::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use gpui::{App, AsyncApp};
use tracing::{debug, error, info, warn};
#[cfg(windows)]
use uds_windows::UnixStream;

#[cfg(windows)]
use self::windows::UnixListener;
use super::ConsoleLog;
use super::command::{self, Outcome};

/// Runs the console commands sent to a Unix domain socket at the path
/// `DUST_CONSOLE_SOCKET` names, if it's set, for as long as the app runs, and
/// sends each connection the messages logged to `log` while it's open. The
/// socket is removed when the app quits.
pub fn serve_console_socket(log: &ConsoleLog, cx: &mut App) {
    let Some(path) = std::env::var_os("DUST_CONSOLE_SOCKET") else {
        return;
    };
    let path = super::user_path(PathBuf::from(path));
    let listener = match listen(&path).and_then(Async::new) {
        Ok(listener) => listener,
        Err(error) => {
            error!("The console can't listen on {}: {error}", path.display());
            return;
        }
    };
    info!("The console listens on {}", path.display());
    cx.on_app_quit(move |_| {
        std::fs::remove_file(&path).ok();
        async {}
    })
    .detach();
    let log = log.clone();
    cx.spawn(async move |cx| accept(listener, log, cx).await)
        .detach();
}

/// Listens at `path`, replacing a socket left there by a game that has exited.
/// Fails if a game is listening there, or if a file other than a socket is in
/// the way.
fn listen(path: &Path) -> io::Result<UnixListener> {
    match UnixListener::bind(path) {
        Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
            if UnixStream::connect(path).is_ok() {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    "another program is listening on it",
                ));
            }
            let metadata = std::fs::symlink_metadata(path)?;
            #[cfg(unix)]
            let is_socket = {
                use std::os::unix::fs::FileTypeExt as _;
                metadata.file_type().is_socket()
            };
            // A Windows AF_UNIX socket is a reparse point.
            #[cfg(windows)]
            let is_socket = {
                use std::os::windows::fs::MetadataExt as _;
                const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
                metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
            };
            if !is_socket {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "a file that isn't a socket is in the way",
                ));
            }
            std::fs::remove_file(path)?;
            UnixListener::bind(path)
        }
        result => result,
    }
}

/// Serves each connection to `listener` in a task of its own, which answers
/// the lines sent on it, and sends the messages logged to `log`, until the
/// client closes it.
async fn accept(listener: Async<UnixListener>, log: ConsoleLog, cx: &mut AsyncApp) {
    loop {
        let accepted = listener
            .read_with(|listener| listener.accept())
            .await
            .and_then(|(stream, _)| Async::new(stream));
        let stream = match accepted {
            Ok(stream) => stream,
            Err(error) => {
                warn!("The console's socket failed to accept a connection: {error}");
                // Such as when out of file descriptors, which a retry right
                // away wouldn't fix.
                Timer::after(Duration::from_millis(100)).await;
                continue;
            }
        };
        // The messages logged from now on.
        let logs = log.messages.activate_cloned();
        cx.spawn(async move |cx| {
            // What to write: the answers to the lines read, and the messages
            // logged, which the `ConsoleLayer` broadcasts as they're logged.
            let (answer_sender, answers) = futures_channel::mpsc::unbounded();
            let stream = &stream;
            let read = async move {
                let mut reader = BufReader::new(stream);
                let mut request = Vec::new();
                loop {
                    request.clear();
                    match reader.read_until(b'\n', &mut request).await {
                        // The client closed the connection.
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(error) => {
                            debug!("A connection to the console's socket failed: {error}");
                            break;
                        }
                    }
                    let line = request.strip_suffix(b"\n").unwrap_or(&request);
                    let line = line.strip_suffix(b"\r").unwrap_or(line);
                    let line = String::from_utf8_lossy(line);
                    let answer = if line.trim().is_empty() {
                        Ok(Outcome::Nothing)
                    } else {
                        command::run(&line, cx).await
                    };
                    let answer = match answer {
                        Ok(Outcome::Output(output)) => ("ok", output),
                        // There's no view here to clear.
                        Ok(Outcome::Nothing | Outcome::Clear) => ("ok", String::new()),
                        Err(error) => ("err", error),
                    };
                    answer_sender.unbounded_send(answer).ok();
                }
                // Ends the answers, and so the writing.
                drop(answer_sender);
            };
            let write = async move {
                let mut writer = stream;
                // A line's log messages arrive before its answer.
                let mut frames = logs.map(|message| ("log", message.to_string())).or(answers);
                while let Some((kind, text)) = frames.next().await {
                    // The length counts the text's bytes, without the line
                    // break after it.
                    let frame = format!("{kind} {}\n{text}\n", text.len());
                    if let Err(error) = writer.write_all(frame.as_bytes()).await {
                        debug!("A connection to the console's socket failed: {error}");
                        return;
                    }
                }
            };
            futures_lite::future::zip(read, write).await;
        })
        .detach();
    }
}

/// `uds_windows`'s listener, which `async_io::Async` can take: it implements
/// `AsSocket`, as `uds_windows`'s own listener doesn't.
#[cfg(windows)]
mod windows {
    use std::io;
    use std::os::windows::io::{AsRawSocket, AsSocket, BorrowedSocket, RawSocket};
    use std::path::Path;

    use uds_windows::{SocketAddr, UnixStream};

    pub struct UnixListener(uds_windows::UnixListener);

    impl UnixListener {
        pub fn bind(path: impl AsRef<Path>) -> io::Result<Self> {
            uds_windows::UnixListener::bind(path).map(Self)
        }

        pub fn accept(&self) -> io::Result<(UnixStream, SocketAddr)> {
            self.0.accept()
        }
    }

    impl AsRawSocket for UnixListener {
        fn as_raw_socket(&self) -> RawSocket {
            self.0.as_raw_socket()
        }
    }

    impl AsSocket for UnixListener {
        fn as_socket(&self) -> BorrowedSocket<'_> {
            // SAFETY: the listener owns the socket, and closes it only when
            // dropped, so it stays open for as long as `self` is borrowed.
            unsafe { BorrowedSocket::borrow_raw(self.0.as_raw_socket()) }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn listen_replaces_a_stale_socket() {
        let dir = std::env::temp_dir().join(format!("dust_console_socket-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("console.sock");
        let _ = std::fs::remove_file(&path);

        let listener = listen(&path).unwrap();
        // Taken while a game listens.
        assert_eq!(listen(&path).unwrap_err().kind(), io::ErrorKind::AddrInUse);
        // Left behind once it's gone.
        drop(listener);
        assert!(path.exists());
        let listener = listen(&path).unwrap();
        drop(listener);

        // Not a socket.
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, "keep me").unwrap();
        assert_eq!(
            listen(&path).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep me");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
