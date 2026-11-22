//! Completing the word before the prompt's cursor on Tab, as a shell does,
//! with clap_complete's engine: the commands, their options, the values of an
//! option or argument that takes one of a few, and paths.
//!
//! Like fish: Tab completes the word if one candidate fits, and otherwise
//! extends it by what the candidates all start with and lists them. While the
//! list is open, Tab and ↓ select the next candidate, putting it in the line,
//! and Shift-Tab and ↑ the one before; Enter keeps the one selected, Escape
//! puts back the word as it was, and any other edit closes the list.

use std::borrow::Cow;
use std::ffi::OsString;
use std::ops::Range;
use std::path::Path;

use clap::CommandFactory as _;

use super::command::CommandLine;

/// A change to the prompt's line: `range` replaced by `text`, which leaves the
/// cursor at the end of the text.
#[derive(Debug, PartialEq)]
pub(super) struct Edit {
    pub range: Range<usize>,
    pub text: String,
}

/// A word the line can take at the cursor.
#[derive(Debug)]
pub(super) struct Candidate {
    /// The word as the command gets it.
    value: String,
    /// The word as the line has it, quoted if it needs to be.
    pub text: String,
    /// What it is, from the command's doc comments.
    pub help: Option<String>,
}

impl Candidate {
    /// The text that replaces the word when the candidate is picked: followed
    /// by a space if it's a whole word, but not a directory, as a file in it
    /// may follow.
    fn picked(&self) -> String {
        if self.value.ends_with(std::path::is_separator) {
            self.text.clone()
        } else {
            format!("{} ", self.text)
        }
    }
}

/// Completes the word before `cursor` in `line`, from the start of the word to
/// the cursor: the edit doing so, if any, and the candidates to list if there
/// are several. A relative path is completed from `directory`.
pub(super) fn complete(
    line: &str,
    cursor: usize,
    directory: Option<&Path>,
) -> (Option<Edit>, Option<Completion>) {
    let Some(words) = words(&line[..cursor]) else {
        return (None, None);
    };
    let (start, typed) = words.last().cloned().expect("ends in the word typed");
    let candidates = candidates(words, directory);
    let word = start..cursor;
    match &candidates[..] {
        [] => (None, None),
        [only] => {
            let edit = Edit {
                range: word,
                text: only.picked(),
            };
            (Some(edit), None)
        }
        [first, rest @ ..] => {
            let common = rest.iter().fold(first.value.as_str(), |common, candidate| {
                common_prefix(common, &candidate.value)
            });
            let edit = (common.len() > typed.len() && common.starts_with(&typed)).then(|| Edit {
                range: word.clone(),
                text: quote(common),
            });
            let mut completion = Completion {
                candidates,
                selected: None,
                line: line.to_owned(),
                word,
                typed: String::new(),
            };
            if let Some(edit) = &edit {
                completion.edit(edit.range.clone(), &edit.text);
            }
            completion.typed = completion.line[completion.word.clone()].to_owned();
            (edit, Some(completion))
        }
    }
}

/// The candidates Tab listed for a word at the prompt, and the one selected.
pub(super) struct Completion {
    candidates: Vec<Candidate>,
    selected: Option<usize>,
    /// The line as the completion last left it, and the word in it that it
    /// completes: the candidate selected, or the word as typed.
    line: String,
    word: Range<usize>,
    /// The word as it was typed, extended by what the candidates all start
    /// with, for Escape to put back.
    typed: String,
}

impl Completion {
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// Whether the prompt is as the completion last left it, `line` with the
    /// cursor at `cursor`. If it isn't, the user changed it since, which ends
    /// the completion.
    pub fn is_current(&self, line: &str, cursor: usize) -> bool {
        self.line == line && self.word.end == cursor
    }

    /// Selects the candidate after the one selected, or before it if not
    /// `forward`, going around, and puts it in the line in place of the word.
    pub fn select_next(&mut self, forward: bool) -> Edit {
        let count = self.candidates.len();
        let index = match (self.selected, forward) {
            (None, true) => 0,
            (None, false) => count - 1,
            (Some(index), true) => (index + 1) % count,
            (Some(index), false) => (index + count - 1) % count,
        };
        self.selected = Some(index);
        let range = self.word.clone();
        let text = self.candidates[index].text.clone();
        self.edit(range.clone(), &text);
        Edit { range, text }
    }

    /// Ends the completion with the candidate selected, followed by a space if
    /// it's a whole word; `None` if none is selected.
    pub fn accept(self) -> Option<Edit> {
        let index = self.selected?;
        Some(self.pick(index))
    }

    /// Ends the completion with candidate `index` in place of the word,
    /// followed by a space if it's a whole word.
    pub fn pick(self, index: usize) -> Edit {
        Edit {
            range: self.word,
            text: self.candidates[index].picked(),
        }
    }

    /// Ends the completion, putting back the word as it was before a candidate
    /// was selected; `None` if none was.
    pub fn cancel(self) -> Option<Edit> {
        self.selected?;
        Some(Edit {
            range: self.word,
            text: self.typed,
        })
    }

    /// Records an edit of the line, which replaces the word.
    fn edit(&mut self, range: Range<usize>, text: &str) {
        self.line.replace_range(range.clone(), text);
        self.word = range.start..range.start + text.len();
    }
}

/// The candidates for the last of `words`, the words of a line up to the
/// cursor and the offsets they start at.
fn candidates(words: Vec<(usize, String)>, directory: Option<&Path>) -> Vec<Candidate> {
    // The engine skips the first word, the program's name. The console's
    // commands are `multicall`, which takes that word for a command, but the
    // engine doesn't know multicall yet: an empty name comes first instead.
    let words: Vec<OsString> = std::iter::once(OsString::new())
        .chain(words.into_iter().map(|(_, word)| word.into()))
        .collect();
    let index = words.len() - 1;
    let mut commands = CommandLine::command();
    let candidates =
        clap_complete::engine::complete(&mut commands, words, index, directory).unwrap_or_default();
    candidates
        .into_iter()
        .filter_map(|candidate| {
            // A path that isn't UTF-8 can't be typed at the prompt.
            let value = candidate.get_value().to_str()?.to_owned();
            Some(Candidate {
                text: quote(&value),
                help: candidate.get_help().map(ToString::to_string),
                value,
            })
        })
        .collect()
}

/// `word` quoted for the line, if it needs to be.
fn quote(word: &str) -> String {
    shlex::try_quote(word).map_or_else(|_| word.to_owned(), Cow::into_owned)
}

/// The longest start `a` and `b` share.
fn common_prefix<'a>(a: &'a str, b: &str) -> &'a str {
    let end = a
        .char_indices()
        .zip(b.chars())
        .find(|&((_, a), b)| a != b)
        .map_or(a.len().min(b.len()), |((index, _), _)| index);
    &a[..end]
}

/// The words of `line` as `shlex::split` takes them, each with the offset it
/// starts at, for completing the last. Unlike `shlex::split`, the last word
/// may be unterminated, inside a quote or after a backslash, as it is while
/// it's typed; and a line that's empty or ends in whitespace ends in an empty
/// word, the next one. `None` if the line ends in a comment, which has nothing
/// to complete.
fn words(line: &str) -> Option<Vec<(usize, String)>> {
    let mut words = Vec::new();
    let mut chars = line.char_indices().peekable();
    loop {
        while chars
            .next_if(|&(_, c)| matches!(c, ' ' | '\t' | '\n'))
            .is_some()
        {}
        let Some(&(start, first)) = chars.peek() else {
            words.push((line.len(), String::new()));
            return Some(words);
        };
        if first == '#' {
            return None;
        }
        let mut word = String::new();
        // The quote the word is inside.
        let mut quote = None;
        loop {
            let Some((_, c)) = chars.next() else {
                words.push((start, word));
                return Some(words);
            };
            match (quote, c) {
                (None, ' ' | '\t' | '\n') => break,
                (None, '\'' | '"') => quote = Some(c),
                (None, '\\') => match chars.next() {
                    Some((_, '\n')) | None => {}
                    Some((_, c)) => word.push(c),
                },
                (Some(open), c) if c == open => quote = None,
                // Inside double quotes, a backslash escapes only these.
                (Some('"'), '\\') => match chars.next() {
                    Some((_, c @ ('$' | '`' | '"' | '\\'))) => word.push(c),
                    Some((_, '\n')) | None => {}
                    Some((_, c)) => {
                        word.push('\\');
                        word.push(c);
                    }
                },
                (_, c) => word.push(c),
            }
        }
        words.push((start, word));
    }
}
