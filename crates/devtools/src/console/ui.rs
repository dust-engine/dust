//! The console tab's view: the toolbar, the message list and the prompt.

use std::collections::VecDeque;
use std::rc::Rc;

use bevy::log::Level;
use bevy::platform::time::Instant;
use gpui::{
    AnyElement, AnyWindowHandle, App, AppContext as _, ClickEvent, Context, Entity, FollowMode,
    HighlightStyle, Hsla, InteractiveElement as _, IntoElement, KeyBinding, ListAlignment,
    ListState, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _,
    Styled as _, StyledText, Subscription, WeakEntity, Window, actions, div, list,
    prelude::FluentBuilder as _, px, rgb,
};
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::input::{Input, InputEvent, InputState, MoveDown, MoveUp};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::{
    ActiveTheme as _, Disableable as _, Icon, IconName, Selectable as _, Sizable as _, Theme,
    h_flex, v_flex,
};

use super::{
    Console, ConsoleEvent, ConsoleMessage, Field, MessageKind, Scope, ValueKind,
    filter_default_allows, submit_line,
};

actions!(dust_devtools_console, [ClearConsole]);

/// Width of the column of icons at the start of each row and of the prompt.
const ICON_WIDTH: f32 = 14.0;

/// Where console events go on the UI thread, a global of the GPUI app: the
/// console view once it exists, a backlog before. Events come from the Bevy
/// thread, and from whichever thread logs.
pub(crate) struct ConsoleSink {
    window: AnyWindowHandle,
    /// When logging started; set by [`super::console_layer`].
    startup: Instant,
    sink: Sink,
}

impl gpui::Global for ConsoleSink {}

enum Sink {
    Backlog(VecDeque<ConsoleEvent>),
    View(Rc<dyn Fn(ConsoleEvent, &mut Window, &mut App)>),
}

/// The most console events kept before the console view exists.
const MAX_BACKLOG: usize = 10_000;

impl ConsoleSink {
    pub fn new(window: AnyWindowHandle) -> Self {
        ConsoleSink {
            window,
            startup: Instant::now(),
            sink: Sink::Backlog(VecDeque::new()),
        }
    }

    /// Passes `event` to the console view, or keeps it until the view exists.
    pub fn send(event: ConsoleEvent, cx: &mut App) {
        let console = cx.global_mut::<ConsoleSink>();
        let sink = match &mut console.sink {
            Sink::Backlog(backlog) => {
                if backlog.len() == MAX_BACKLOG {
                    backlog.pop_front();
                }
                backlog.push_back(event);
                return;
            }
            Sink::View(sink) => sink.clone(),
        };
        let window = console.window;
        window
            .update(cx, |_, window, cx| sink(event, window, cx))
            .ok();
    }

    /// Passes the console events to `sink` from now on, after those that
    /// arrived before.
    pub fn attach(sink: impl Fn(ConsoleEvent, &mut Window, &mut App) + 'static, cx: &mut App) {
        let sink: Rc<dyn Fn(ConsoleEvent, &mut Window, &mut App)> = Rc::new(sink);
        let console = cx.global_mut::<ConsoleSink>();
        let window = console.window;
        let previous = std::mem::replace(&mut console.sink, Sink::View(sink.clone()));
        if let Sink::Backlog(backlog) = previous {
            window
                .update(cx, |_, window, cx| {
                    for event in backlog {
                        sink(event, window, cx);
                    }
                })
                .ok();
        }
    }

    pub fn set_startup(startup: Instant, cx: &mut App) {
        cx.global_mut::<ConsoleSink>().startup = startup;
    }

    pub fn startup(cx: &App) -> Instant {
        cx.global::<ConsoleSink>().startup
    }
}

pub(crate) struct ConsoleView {
    console: Console,
    filter: Entity<InputState>,
    prompt: Entity<InputState>,
    list: ListState,
    /// The `serial`s of the rows the list shows, in order.
    shown: Vec<u64>,
    /// The count of the last shown row when the list was last updated.
    last_count: u32,
    /// The rows changed since the list was last updated.
    dirty: bool,
    startup: Instant,
    _subscriptions: Vec<Subscription>,
}

impl ConsoleView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.bind_keys([KeyBinding::new("cmd-k", ClearConsole, Some("Console"))]);
        let filter = cx.new(|cx| InputState::new(window, cx).placeholder("Filter"));
        let prompt =
            cx.new(|cx| InputState::new(window, cx).placeholder("help lists the commands"));
        let subscriptions = vec![
            cx.subscribe(&filter, |this, filter, event: &InputEvent, cx| {
                if let InputEvent::Change = event {
                    this.console.filter = filter.read(cx).value().to_string();
                    this.dirty = true;
                    cx.notify();
                }
            }),
            cx.subscribe_in(
                &prompt,
                window,
                |this, prompt, event: &InputEvent, window, cx| {
                    if let InputEvent::PressEnter { .. } = event {
                        this.console.prompt.line = prompt.read(cx).value().to_string();
                        if let Some(line) = this.console.submit() {
                            submit_line(cx, line);
                        }
                        prompt.update(cx, |prompt, cx| prompt.set_value("", window, cx));
                        this.list.scroll_to_end();
                        this.dirty = true;
                        cx.notify();
                    }
                },
            ),
        ];

        // Attached once this view exists, as the backlog is delivered at once.
        let view = cx.weak_entity();
        cx.defer(move |cx| {
            ConsoleSink::attach(
                move |event, _, cx| {
                    if let ConsoleEvent::Dock(side) = event {
                        crate::devtools::set_dock(side, cx);
                        return;
                    }
                    view.update(cx, |view, cx| {
                        view.console.apply(event);
                        view.dirty = true;
                        cx.notify();
                    })
                    .ok();
                },
                cx,
            );
        });

        let list = ListState::new(0, ListAlignment::Top, px(200.));
        list.set_follow_mode(FollowMode::Tail);
        ConsoleView {
            console: Console::default(),
            filter,
            prompt,
            list,
            shown: Vec::new(),
            last_count: 0,
            dirty: false,
            startup: ConsoleSink::startup(cx),
            _subscriptions: subscriptions,
        }
    }

    /// Brings the list up to date with the rows the scope bar and the filter
    /// field let through. Rows only come and go at the ends while nothing else
    /// changes, which keeps the measured heights of the others.
    fn update_list(&mut self) {
        if !std::mem::take(&mut self.dirty) {
            return;
        }
        let filter = self.console.filter.trim();
        let shown: Vec<u64> = self
            .console
            .rows
            .iter()
            .filter(|row| self.console.shows(&row.message, filter))
            .map(|row| row.serial)
            .collect();
        let dropped = match shown.first() {
            Some(&first) => self
                .shown
                .iter()
                .take_while(|&&serial| serial < first)
                .count(),
            None => self.shown.len(),
        };
        let kept = &self.shown[dropped..];
        if shown.starts_with(kept) {
            self.list.splice(0..dropped, 0);
            self.list
                .splice(kept.len()..kept.len(), shown.len() - kept.len());
            // A repeat only increments the last row's count.
            let last_count = self.console.rows.back().map_or(0, |row| row.count);
            if !kept.is_empty() && last_count != self.last_count {
                self.list.remeasure_items(kept.len() - 1..kept.len());
            }
        } else {
            self.list.reset(shown.len());
        }
        self.last_count = self.console.rows.back().map_or(0, |row| row.count);
        self.shown = shown;
        if std::mem::take(&mut self.console.scroll_to_bottom) {
            self.list.scroll_to_end();
        }
    }

    fn toggle_expanded(&mut self, ix: usize, cx: &mut Context<Self>) {
        let serial = self.shown[ix];
        if let Some(row) = self
            .console
            .rows
            .iter_mut()
            .find(|row| row.serial == serial)
        {
            row.expanded = !row.expanded;
            self.list.remeasure_items(ix..ix + 1);
            cx.notify();
        }
    }

    /// The filter field, the scope bar, and the button clearing the console.
    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        h_flex()
            .gap_1()
            .p_1()
            .flex_wrap()
            .border_b_1()
            .border_color(theme.border)
            .child(
                div()
                    .w(px(140.))
                    .child(Input::new(&self.filter).xsmall().cleanable(true)),
            )
            .child(self.scope_bar(cx))
            .child(div().flex_1())
            .child(
                Button::new("console-clear")
                    .icon(IconName::Delete)
                    .ghost()
                    .xsmall()
                    .tooltip("Clear Console (⌘K)")
                    .on_click(
                        cx.listener(|this, _, window, cx| this.clear(&ClearConsole, window, cx)),
                    ),
            )
    }

    fn clear(&mut self, _: &ClearConsole, _: &mut Window, cx: &mut Context<Self>) {
        self.console.rows.clear();
        self.dirty = true;
        cx.notify();
    }

    /// Safari's scope bar: "All", or only the picked scopes. A click picks one
    /// scope; a click with Cmd held adds or removes one.
    ///
    /// The scope of a level `LogPlugin`'s filter drops is disabled, with a
    /// tooltip saying how to let the level through, until a message of that
    /// level arrives anyway, from a target with a directive of its own.
    fn scope_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let picked = self.console.scopes;
        let all = !picked.contains(&true);
        let mut bar = h_flex().gap_0p5().child(
            Button::new("console-scope-all")
                .label("All")
                .ghost()
                .xsmall()
                .selected(all)
                .on_click(cx.listener(|this, _, _, cx| {
                    this.console.scopes = Default::default();
                    this.dirty = true;
                    cx.notify();
                })),
        );
        for scope in Scope::ALL {
            let index = scope as usize;
            let filtered_out = scope
                .level()
                .filter(|&level| !self.console.received[index] && !filter_default_allows(level));
            let tooltip = match filtered_out {
                Some(level) => {
                    let level = level.as_str().to_lowercase();
                    format!(
                        "LogPlugin filters out {level} messages. \
                         Restart with RUST_LOG={level} to show them."
                    )
                }
                None => "⌘-click to show several".to_owned(),
            };
            bar = bar.child(
                Button::new(("console-scope", index))
                    .label(scope.label())
                    .ghost()
                    .xsmall()
                    .selected(picked[index])
                    .disabled(filtered_out.is_some())
                    .tooltip(tooltip)
                    .on_click(cx.listener(move |this, event: &ClickEvent, _, cx| {
                        let scopes = &mut this.console.scopes;
                        if event.modifiers().platform {
                            scopes[index] = !scopes[index];
                        } else {
                            *scopes = Default::default();
                            scopes[index] = true;
                        }
                        this.dirty = true;
                        cx.notify();
                    })),
            );
        }
        bar
    }

    /// The command line: Enter submits it, ↑ and ↓ recall earlier lines.
    fn prompt(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        h_flex()
            .gap_1()
            .px_1()
            .border_t_1()
            .border_color(theme.border)
            // A single-line input leaves ↑ and ↓ to its ancestors.
            .on_action(cx.listener(|this, _: &MoveUp, window, cx| this.browse(true, window, cx)))
            .on_action(cx.listener(|this, _: &MoveDown, window, cx| this.browse(false, window, cx)))
            .child(
                Icon::new(IconName::ChevronRight)
                    .xsmall()
                    .text_color(theme.primary),
            )
            .child(
                div()
                    .flex_1()
                    .font_family(theme.mono_font_family.clone())
                    .child(Input::new(&self.prompt).appearance(false).small()),
            )
    }

    fn browse(&mut self, older: bool, window: &mut Window, cx: &mut Context<Self>) {
        let prompt = &mut self.console.prompt;
        prompt.line = self.prompt.read(cx).value().to_string();
        prompt.browse(older);
        let line = prompt.line.clone();
        self.prompt
            .update(cx, |prompt, cx| prompt.set_value(line, window, cx));
    }

    /// A row as in Safari: an icon for the kind of message, the repeat count,
    /// a triangle expanding the details, the message with its fields, and the
    /// target it was logged to on the right.
    fn row(&self, ix: usize, view: WeakEntity<Self>, cx: &App) -> AnyElement {
        let Some(row) = self
            .shown
            .get(ix)
            .and_then(|&serial| self.console.rows.iter().find(|row| row.serial == serial))
        else {
            return div().into_any_element();
        };
        let theme = cx.theme();
        let message = &row.message;
        let fill = match message.kind {
            MessageKind::Log(Level::ERROR) | MessageKind::CommandError => {
                theme.danger.opacity(0.15)
            }
            MessageKind::Log(Level::WARN) => theme.warning.opacity(0.15),
            _ => gpui::transparent_black(),
        };
        let expandable = message.source.is_some();
        let mut summary = v_flex().flex_1().min_w_0().child(summary(message, theme));
        if row.expanded {
            summary = summary.children(details(message, self.startup, theme));
        }
        h_flex()
            .id(("console-row", row.serial as usize))
            .items_start()
            .gap_1()
            .px_1p5()
            .py_0p5()
            .bg(fill)
            .border_b_1()
            .border_color(theme.border)
            .text_sm()
            // Commands and their output in monospace, so tables like clap's
            // help line up.
            .when(!expandable, |row| {
                row.font_family(theme.mono_font_family.clone())
            })
            .child(
                div()
                    .w(px(ICON_WIDTH))
                    .flex_shrink_0()
                    .child(icon(message.kind, theme)),
            )
            .when(row.count > 1, |this| {
                this.child(count_badge(row.count, message.kind, theme))
            })
            .when(expandable, |this| {
                this.child(
                    div()
                        .id(("console-disclosure", row.serial as usize))
                        .flex_shrink_0()
                        .cursor_pointer()
                        .on_click(move |_, _, cx| {
                            view.update(cx, |view, cx| view.toggle_expanded(ix, cx))
                                .ok();
                        })
                        .child(
                            Icon::new(if row.expanded {
                                IconName::ChevronDown
                            } else {
                                IconName::ChevronRight
                            })
                            .xsmall()
                            .text_color(theme.muted_foreground),
                        ),
                )
            })
            .child(summary)
            .children(message.source.as_ref().map(|source| {
                div()
                    .flex_shrink_0()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(source.target.clone()))
            }))
            .into_any_element()
    }
}

impl Render for ConsoleView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.update_list();
        let view = cx.weak_entity();
        let rows = list(self.list.clone(), move |ix, _, cx| {
            let Some(entity) = view.upgrade() else {
                return div().into_any_element();
            };
            entity.read(cx).row(ix, view.clone(), cx)
        })
        .size_full();
        v_flex()
            .key_context("Console")
            .on_action(cx.listener(Self::clear))
            .size_full()
            .child(self.toolbar(cx))
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .child(rows)
                    .vertical_scrollbar(&self.list),
            )
            .child(self.prompt(cx))
    }
}

/// The message followed by its fields as `name=value`, like the log lines on
/// stderr.
fn summary(message: &ConsoleMessage, theme: &Theme) -> StyledText {
    let color = match message.kind {
        MessageKind::Log(Level::ERROR) | MessageKind::CommandError => theme.danger,
        MessageKind::Log(Level::WARN) => theme.warning,
        MessageKind::Log(Level::TRACE) => theme.muted_foreground,
        _ => theme.foreground,
    };
    let mut text = Highlighted::default();
    text.push(&message.text, color);
    for field in &message.fields {
        if !text.text.is_empty() {
            text.push("  ", color);
        }
        text.push(&format!("{}=", field.name), theme.muted_foreground);
        push_value(&mut text, field, theme);
    }
    text.build()
}

/// The fields one per line, the spans the message was logged in, where it
/// was logged, and when.
fn details(message: &ConsoleMessage, startup: Instant, theme: &Theme) -> Vec<StyledText> {
    let weak = theme.muted_foreground;
    let normal = theme.foreground;
    let line = |name: &str| {
        let mut text = Highlighted::default();
        text.push(&format!("{name}: "), weak);
        text
    };

    let mut lines = Vec::new();
    for field in &message.fields {
        let mut text = line(field.name);
        push_value(&mut text, field, theme);
        lines.push(text);
    }
    for span in &message.spans {
        let mut text = line("span");
        text.push(span.name, normal);
        if !span.fields.is_empty() {
            text.push("{", weak);
            for (i, field) in span.fields.iter().enumerate() {
                if i > 0 {
                    text.push(" ", weak);
                }
                text.push(&format!("{}=", field.name), weak);
                push_value(&mut text, field, theme);
            }
            text.push("}", weak);
        }
        lines.push(text);
    }
    let mut plain = Vec::new();
    if let Some(source) = &message.source {
        plain.push(("target", source.target.clone()));
        if let Some(module_path) = &source.module_path
            && *module_path != source.target
        {
            plain.push(("module", module_path.clone()));
        }
        if let Some(file) = &source.file {
            let location = match source.line {
                Some(number) => format!("{file}:{number}"),
                None => file.clone(),
            };
            plain.push(("location", location));
        }
        if let Some(thread) = &source.thread {
            plain.push(("thread", thread.clone()));
        }
    }
    let seconds = message
        .time
        .saturating_duration_since(startup)
        .as_secs_f64();
    plain.push(("time", format!("{seconds:.3} s after startup")));
    for (name, value) in plain {
        let mut text = line(name);
        text.push(&value, normal);
        lines.push(text);
    }
    lines.into_iter().map(Highlighted::build).collect()
}

/// Appends a field's value, colored by type as in Xcode: strings red and
/// quoted, numbers yellow, booleans pink, errors in the error color.
fn push_value(text: &mut Highlighted, field: &Field, theme: &Theme) {
    match field.kind {
        ValueKind::Str => text.push(&format!("{:?}", field.value), rgb(0xfc6a5d).into()),
        ValueKind::Number => text.push(&field.value, rgb(0xd0bf69).into()),
        ValueKind::Bool => text.push(&field.value, rgb(0xfc5fa3).into()),
        ValueKind::Error => text.push(&field.value, theme.danger),
        ValueKind::Debug => text.push(&field.value, theme.foreground),
    }
}

/// Text built from colored pieces.
#[derive(Default)]
struct Highlighted {
    text: String,
    highlights: Vec<(std::ops::Range<usize>, HighlightStyle)>,
}

impl Highlighted {
    fn push(&mut self, piece: &str, color: Hsla) {
        let start = self.text.len();
        self.text.push_str(piece);
        self.highlights.push((
            start..self.text.len(),
            HighlightStyle {
                color: Some(color),
                ..Default::default()
            },
        ));
    }

    fn build(self) -> StyledText {
        StyledText::new(self.text).with_highlights(self.highlights)
    }
}

/// Safari's repeat count: a pill with the number of times the message
/// arrived in a row.
fn count_badge(count: u32, kind: MessageKind, theme: &Theme) -> impl IntoElement {
    let (fill, text_color): (Hsla, Hsla) = match kind {
        MessageKind::Log(Level::ERROR) | MessageKind::CommandError => (theme.danger, gpui::white()),
        MessageKind::Log(Level::WARN) => (theme.warning, gpui::black()),
        _ => (theme.info, gpui::white()),
    };
    div()
        .flex_shrink_0()
        .px_1()
        .min_w(px(14.))
        .rounded_full()
        .bg(fill)
        .text_color(text_color)
        .text_xs()
        .text_center()
        .child(count.to_string())
}

/// The icons follow Safari's: errors, warnings and infos by their usual
/// symbols, ">" for a submitted line and "<" for what it printed.
fn icon(kind: MessageKind, theme: &Theme) -> AnyElement {
    let icon = |name: IconName, color: Hsla| {
        Icon::new(name)
            .xsmall()
            .text_color(color)
            .into_any_element()
    };
    match kind {
        MessageKind::Log(Level::ERROR) | MessageKind::CommandError => {
            icon(IconName::CircleX, theme.danger)
        }
        MessageKind::Log(Level::WARN) => icon(IconName::TriangleAlert, theme.warning),
        MessageKind::Log(Level::INFO) => icon(IconName::Info, theme.info),
        MessageKind::Log(Level::DEBUG) => icon(IconName::Ellipsis, theme.foreground),
        MessageKind::Log(_) => div()
            .mt(px(6.))
            .ml(px(4.))
            .size(px(3.))
            .rounded_full()
            .bg(theme.muted_foreground)
            .into_any_element(),
        MessageKind::Command => icon(IconName::ChevronRight, theme.primary),
        MessageKind::Output => div()
            .text_color(theme.muted_foreground)
            .child("‹")
            .into_any_element(),
    }
}
