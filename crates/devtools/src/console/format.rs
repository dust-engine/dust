//! A console message as text, as the console's socket sends it.

use std::fmt::{self, Write as _};

use super::{ConsoleMessage, Field, MessageKind, ValueKind};

/// A log message as the `fmt` layer prints it to stderr, after the seconds
/// since logging started:
///
/// ```text
/// 3.456  WARN load{model="castle"}: dust_vox: loaded in 12 ms voxels=1024
/// ```
///
/// The format is for log messages: a command line or its output prints with
/// no level.
impl fmt::Display for ConsoleMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let level = match self.kind {
            MessageKind::Log(level) => level.as_str(),
            _ => "",
        };
        write!(f, "{:.3} {level:>5} ", self.time.as_secs_f64())?;
        for span in &self.spans {
            f.write_str(span.name)?;
            if !span.fields.is_empty() {
                f.write_char('{')?;
                for (i, field) in span.fields.iter().enumerate() {
                    if i > 0 {
                        f.write_char(' ')?;
                    }
                    write!(f, "{field}")?;
                }
                f.write_char('}')?;
            }
            f.write_char(':')?;
        }
        if !self.spans.is_empty() {
            f.write_char(' ')?;
        }
        if let Some(source) = &self.source {
            write!(f, "{}: ", source.target)?;
        }
        f.write_str(&self.text)?;
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 || !self.text.is_empty() {
                f.write_char(' ')?;
            }
            write!(f, "{field}")?;
        }
        Ok(())
    }
}

/// `name=value`, with strings and errors quoted.
impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ValueKind::Str | ValueKind::Error => write!(f, "{}={:?}", self.name, self.value),
            ValueKind::Number | ValueKind::Bool | ValueKind::Debug => {
                write!(f, "{}={}", self.name, self.value)
            }
        }
    }
}
