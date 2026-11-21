//! `dustctl` runs commands in the DevTools console of a running Dust game,
//! through the socket its console listens on when the game is started with
//! `DUST_CONSOLE_SOCKET` set. The console's `remote.rs` describes the
//! protocol.

use std::io::{self, BufRead, BufReader, IsTerminal as _, Write as _};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use clap::Parser;
#[cfg(windows)]
use uds_windows::UnixStream;

/// The environment variable naming the socket, which the game listens on
/// only when started with it set.
const SOCKET_ENV: &str = "DUST_CONSOLE_SOCKET";

/// Runs commands in the DevTools console of a running Dust game.
///
/// Start the game with DUST_CONSOLE_SOCKET set to a path for its console to
/// listen on, and give dustctl the same path, in that variable or with
/// --socket. `dustctl help` lists the console's commands.
///
/// What a command prints goes to standard output; why it failed, to standard
/// error, and dustctl then exits with 1. The messages the game logs while
/// dustctl is connected go to standard error as they arrive; the game's own
/// standard error has the ones before. At a terminal, without a command,
/// dustctl reads lines at a prompt, and shows the messages among the answers.
#[derive(Parser)]
#[command(name = "dustctl")]
struct Args {
    /// The socket the game's console listens on
    #[arg(long, env = SOCKET_ENV, value_name = "PATH")]
    socket: PathBuf,
    /// How long to wait for each answer, in seconds
    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    timeout: u64,
    /// The console line, as typed at its prompt: a command and its arguments,
    /// such as `screenshot shot.png`, or all of it as a single argument.
    /// Without one, each line of standard input runs in turn.
    #[arg(
        trailing_var_arg = true,
        allow_hyphen_values = true,
        value_name = "COMMAND"
    )]
    command: Vec<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    // Under `bazel run`, which runs dustctl in its runfiles, a relative path
    // is taken from the directory `bazel run` was started in, as the user
    // meant.
    let path = match std::env::var_os("BUILD_WORKING_DIRECTORY") {
        Some(dir) if args.socket.is_relative() => Path::new(&dir).join(&args.socket),
        _ => args.socket.clone(),
    };
    match run(&args, &path) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("dustctl: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Runs the console lines `args` gives, printing what each printed; whether
/// they all succeeded.
fn run(args: &Args, path: &Path) -> Result<bool, String> {
    let stream = UnixStream::connect(path).map_err(|error| {
        let path = path.display();
        match error.kind() {
            io::ErrorKind::NotFound => format!(
                "no socket at {path}; start the game with {SOCKET_ENV}={path} for its \
                 console to listen there"
            ),
            io::ErrorKind::ConnectionRefused => format!(
                "no game listens on {path}; the socket there is left from one that has exited"
            ),
            _ => format!("can't connect to {path}: {error}"),
        }
    })?;
    let interactive = args.command.is_empty() && io::stdin().is_terminal();
    // Whether the prompt is waiting for a line. A log message printed then
    // goes over the prompt, and shows it again below.
    let prompting = Arc::new(AtomicBool::new(false));

    // Reads what the game sends: the answers, for `Console::run`, and the log
    // messages, which it prints: at the prompt, among the answers, and
    // otherwise to standard error, which keeps standard output for what the
    // commands print.
    let (answer_sender, answers) = mpsc::channel();
    let mut reader = BufReader::new(stream.try_clone().map_err(|error| error.to_string())?);
    let reading_prompt = prompting.clone();
    std::thread::spawn(move || {
        loop {
            match read_frame(&mut reader) {
                Ok((kind, text)) if kind == "log" && !interactive => eprintln!("{text}"),
                Ok((kind, text)) if kind == "log" => {
                    if reading_prompt.load(Ordering::Relaxed) {
                        // Over the prompt, which then shows again below.
                        println!("\r{text}");
                        eprint!("dust> ");
                    } else {
                        println!("{text}");
                    }
                }
                frame => {
                    let failed = frame.is_err();
                    if answer_sender.send(frame).is_err() || failed {
                        return;
                    }
                }
            }
        }
    });

    let mut console = Console {
        writer: &stream,
        answers,
        timeout: args.timeout,
    };
    match &args.command[..] {
        [] => {}
        // A single word with whitespace is the whole line already.
        [line] if line.contains(char::is_whitespace) => return console.run(line),
        // Quoted as needed, so that the console splits the line back into
        // the same words.
        words => {
            let line = shlex::try_join(words.iter().map(String::as_str))
                .map_err(|_| "a console line can't hold a NUL character".to_owned())?;
            return console.run(&line);
        }
    }

    let stdin = io::stdin();
    let mut succeeded = true;
    loop {
        if interactive {
            eprint!("dust> ");
            io::stderr().flush().ok();
            prompting.store(true, Ordering::Relaxed);
        }
        let mut line = String::new();
        let read = stdin.lock().read_line(&mut line);
        prompting.store(false, Ordering::Relaxed);
        if read.map_err(|error| error.to_string())? == 0 {
            return Ok(succeeded);
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if !line.trim().is_empty() {
            succeeded &= console.run(line)?;
        }
    }
}

/// Reads what the game sends next: `ok`, `err` or `log`, a space and the
/// text's length in bytes, a line break, then the text and a line break.
/// Returns the first word and the text.
fn read_frame(reader: &mut impl BufRead) -> Result<(String, String), String> {
    let closed = || "the game closed the connection".to_owned();
    let mut header = String::new();
    if reader
        .read_line(&mut header)
        .map_err(|error| error.to_string())?
        == 0
    {
        return Err(closed());
    }
    let invalid = || format!("the game sent {header:?}");
    let (kind, length) = header.trim_end().split_once(' ').ok_or_else(invalid)?;
    let length: usize = length.parse().map_err(|_| invalid())?;
    let mut text = vec![0; length + 1];
    reader
        .read_exact(&mut text)
        .map_err(|error| match error.kind() {
            io::ErrorKind::UnexpectedEof => closed(),
            _ => error.to_string(),
        })?;
    if text.pop() != Some(b'\n') {
        return Err("the game sent more than it said it would".to_owned());
    }
    let text =
        String::from_utf8(text).map_err(|_| "the game sent text that isn't UTF-8".to_owned())?;
    Ok((kind.to_owned(), text))
}

/// A connection to the game's console.
struct Console<'a> {
    writer: &'a UnixStream,
    /// The answers to the lines sent, from the thread reading, or why it
    /// stopped.
    answers: Receiver<Result<(String, String), String>>,
    timeout: u64,
}

impl Console<'_> {
    /// Runs `line` and prints what it printed, or why it failed; whether it
    /// succeeded.
    fn run(&mut self, line: &str) -> Result<bool, String> {
        // A line break would end the line early.
        if line.contains(['\n', '\r']) {
            return Err("a console line can't hold a line break".to_owned());
        }
        writeln!(self.writer, "{line}")
            .and_then(|()| self.writer.flush())
            .map_err(|error| error.to_string())?;
        let (status, output) = match self.answers.recv_timeout(Duration::from_secs(self.timeout)) {
            Ok(answer) => answer?,
            Err(RecvTimeoutError::Timeout) => {
                return Err(format!(
                    "the game didn't answer in {} s; it runs no frames while its window is \
                     hidden or the screen is locked, and --timeout waits longer",
                    self.timeout
                ));
            }
            // The thread reading stops after sending why, which came before.
            Err(RecvTimeoutError::Disconnected) => {
                return Err("the game closed the connection".to_owned());
            }
        };
        match status.as_str() {
            "ok" => {
                if !output.is_empty() {
                    println!("{output}");
                }
                Ok(true)
            }
            "err" => {
                eprintln!("{output}");
                Ok(false)
            }
            _ => Err(format!("the game answered {status:?}")),
        }
    }
}
