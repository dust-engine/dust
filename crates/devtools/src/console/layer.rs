//! The `tracing` layer that copies log messages into the console.

use std::fmt::{self, Write as _};

use bevy::platform::time::Instant;
use futures_channel::mpsc::UnboundedSender;
use tracing::field::{self as tracing_field, Visit};
use tracing::{Event, Subscriber, span};
use tracing_log::NormalizeEvent;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

use super::{ConsoleEvent, ConsoleMessage, Field, MessageKind, Source, SpanContext, ValueKind};

/// The `tracing` layer that copies log messages into the DevTools console;
/// made by [`console_layer`](super::console_layer).
///
/// It sends each message on from whichever thread logged it, to the
/// [`ConsoleLog`](super::ConsoleLog) made with it.
pub struct ConsoleLayer {
    pub(super) events: UnboundedSender<ConsoleEvent>,
}

/// A span's fields, kept in the span's extensions for the events logged in
/// it.
struct SpanFields(Vec<Field>);

impl<S> Layer<S> for ConsoleLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let mut fields = FieldVisitor::default();
        attrs.record(&mut fields);
        if !fields.0.is_empty()
            && let Some(span) = ctx.span(id)
        {
            span.extensions_mut().insert(SpanFields(fields.0));
        }
    }

    /// Fields can be given a value after the span is created, with
    /// `Span::record`.
    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut recorded = FieldVisitor::default();
        values.record(&mut recorded);
        let mut extensions = span.extensions_mut();
        match extensions.get_mut::<SpanFields>() {
            Some(SpanFields(fields)) => {
                for field in recorded.0 {
                    match fields.iter_mut().find(|old| old.name == field.name) {
                        Some(old) => *old = field,
                        None => fields.push(field),
                    }
                }
            }
            None => extensions.insert(SpanFields(recorded.0)),
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        // `tracing-log` logs all `log` records from one callsite, keeping each
        // record's own target, module path, file and line in `log.*` fields.
        // `normalized_metadata` puts them back into a `Metadata`.
        let normalized = event.normalized_metadata();
        let metadata = normalized.as_ref().unwrap_or(event.metadata());
        let mut fields = FieldVisitor::default();
        event.record(&mut fields);
        let mut fields = fields.0;
        if event.is_log() {
            fields.retain(|field| !field.name.starts_with("log."));
        }
        let text = match fields.iter().position(|field| field.name == "message") {
            Some(index) => fields.remove(index).value,
            None => String::new(),
        };
        let spans = ctx.event_scope(event).map_or_else(Vec::new, |scope| {
            scope
                .from_root()
                .map(|span| SpanContext {
                    name: span.name(),
                    fields: span
                        .extensions()
                        .get::<SpanFields>()
                        .map_or_else(Vec::new, |fields| fields.0.clone()),
                })
                .collect()
        });
        let message = ConsoleMessage {
            kind: MessageKind::Log(*metadata.level()),
            text,
            fields,
            spans,
            source: Some(Source {
                target: metadata.target().to_owned(),
                module_path: metadata.module_path().map(str::to_owned),
                file: metadata.file().map(str::to_owned),
                line: metadata.line(),
                thread: std::thread::current().name().map(str::to_owned),
            }),
            time: Instant::now(),
        };
        // Fails only once the console is gone.
        self.events
            .unbounded_send(ConsoleEvent::Message(message))
            .ok();
    }
}

#[derive(Default)]
struct FieldVisitor(Vec<Field>);

impl FieldVisitor {
    fn push(&mut self, field: &tracing_field::Field, value: String, kind: ValueKind) {
        self.0.push(Field {
            name: field.name(),
            value,
            kind,
        });
    }
}

impl Visit for FieldVisitor {
    fn record_str(&mut self, field: &tracing_field::Field, value: &str) {
        self.push(field, value.to_owned(), ValueKind::Str);
    }

    fn record_bool(&mut self, field: &tracing_field::Field, value: bool) {
        self.push(field, value.to_string(), ValueKind::Bool);
    }

    fn record_i64(&mut self, field: &tracing_field::Field, value: i64) {
        self.push(field, value.to_string(), ValueKind::Number);
    }

    fn record_u64(&mut self, field: &tracing_field::Field, value: u64) {
        self.push(field, value.to_string(), ValueKind::Number);
    }

    fn record_i128(&mut self, field: &tracing_field::Field, value: i128) {
        self.push(field, value.to_string(), ValueKind::Number);
    }

    fn record_u128(&mut self, field: &tracing_field::Field, value: u128) {
        self.push(field, value.to_string(), ValueKind::Number);
    }

    fn record_f64(&mut self, field: &tracing_field::Field, value: f64) {
        self.push(field, value.to_string(), ValueKind::Number);
    }

    /// Writes the error and its sources as `error: source: source's source`.
    fn record_error(
        &mut self,
        field: &tracing_field::Field,
        value: &(dyn std::error::Error + 'static),
    ) {
        let mut text = value.to_string();
        let mut source = value.source();
        while let Some(error) = source {
            let _ = write!(text, ": {error}");
            source = error.source();
        }
        self.push(field, text, ValueKind::Error);
    }

    /// Also receives the `message` of `info!("…")` and the like, as the
    /// `fmt::Arguments` of the format string.
    fn record_debug(&mut self, field: &tracing_field::Field, value: &dyn fmt::Debug) {
        self.push(field, format!("{value:?}"), ValueKind::Debug);
    }
}
