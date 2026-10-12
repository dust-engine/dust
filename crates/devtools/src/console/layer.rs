//! The `tracing` layer that copies log messages into the console.

use std::fmt::{self, Write as _};
use std::sync::mpsc;

use bevy::log::BoxedLayer;
use bevy::log::tracing::field::{self as tracing_field, Visit};
use bevy::log::tracing::{Event, Subscriber, span};
use bevy::log::tracing_subscriber::Layer;
use bevy::log::tracing_subscriber::layer::Context;
use bevy::log::tracing_subscriber::registry::LookupSpan;
use bevy::platform::cell::SyncCell;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use tracing_log::NormalizeEvent;

use super::{Console, ConsoleMessage, Field, MessageKind, Source, SpanContext, ValueKind};

/// Copies log messages into the DevTools console. Install it as
/// `LogPlugin::custom_layer`:
///
/// ```ignore
/// app.add_plugins(DefaultPlugins.set(LogPlugin {
///     custom_layer: dust_devtools::console_layer,
///     ..default()
/// }));
/// ```
///
/// The console gets what passes `LogPlugin`'s filter, the same messages it
/// prints to stderr: `tracing` events, and `log` records, which `LogPlugin`
/// turns into `tracing` events.
pub fn console_layer(app: &mut App) -> Option<BoxedLayer> {
    let (sender, receiver) = mpsc::channel();
    app.init_resource::<Console>();
    app.insert_resource(LogReceiver(SyncCell::new(receiver)));
    app.add_systems(Update, receive_log_messages);
    Some(Box::new(ConsoleLayer { sender }))
}

/// The receiving end of [`ConsoleLayer`]'s channel. The layer runs on whichever
/// thread logs, so messages go through the channel and are moved into
/// [`Console`] once per frame.
#[derive(Resource)]
pub(super) struct LogReceiver(SyncCell<mpsc::Receiver<ConsoleMessage>>);

pub(super) fn receive_log_messages(
    mut receiver: ResMut<LogReceiver>,
    mut console: ResMut<Console>,
) {
    for message in receiver.0.get().try_iter() {
        console.push(message);
    }
}

pub(super) struct ConsoleLayer {
    pub(super) sender: mpsc::Sender<ConsoleMessage>,
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
        // Sending fails only once the app, and with it the receiver, is gone.
        let _ = self.sender.send(message);
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
