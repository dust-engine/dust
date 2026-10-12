//! Drawing of the console tab: the toolbar, the message list and the prompt.

use bevy::log::Level;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use pumicite_egui::egui;

use super::{
    Console, ConsoleMessage, Field, MessageKind, Row, Scope, ValueKind, filter_default_allows,
};
use crate::devtools::icon_button;

const CLEAR_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::K);

/// Width of the column of icons at the start of each row and of the prompt.
const ICON_WIDTH: f32 = 14.0;

const ROW_MARGIN: egui::Margin = egui::Margin::symmetric(6, 3);

const INFO_COLOR: egui::Color32 = egui::Color32::from_rgb(0x4c, 0x9a, 0xff);

pub(super) fn console_ui(
    InMut(ui): InMut<egui::Ui>,
    mut console: ResMut<Console>,
    time: Res<Time<Real>>,
) {
    let console = &mut *console;
    toolbar(ui, console);
    egui::Panel::bottom("devtools_console_prompt").show(ui, |ui| prompt(ui, console));
    message_list(ui, console, time.startup());
}

/// The filter field, the scope bar, and the button clearing the console.
fn toolbar(ui: &mut egui::Ui, console: &mut Console) {
    let is_mac = ui.ctx().os().is_mac();
    ui.horizontal(|ui| {
        // Right-to-left first, so the clear button keeps its place at the end
        // and the scope bar wraps in the width left.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let shortcut = CLEAR_SHORTCUT.format(&egui::ModifierNames::NAMES, is_mac);
            let clear =
                icon_button(ui, paint_trash).on_hover_text(format!("Clear Console ({shortcut})"));
            if clear.clicked() || ui.input_mut(|input| input.consume_shortcut(&CLEAR_SHORTCUT)) {
                console.rows.clear();
            }
            let layout = egui::Layout::left_to_right(egui::Align::Center).with_main_wrap(true);
            ui.with_layout(layout, |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut console.filter)
                        .hint_text("Filter")
                        .desired_width(120.0),
                );
                scope_bar(ui, &mut console.scopes, &console.received, is_mac);
            });
        });
    });
    ui.separator();
}

/// Safari's scope bar: "All", or only the picked scopes. A click picks one
/// scope; a click with Cmd (Ctrl) held adds or removes one.
///
/// The scope of a level `LogPlugin`'s filter drops is disabled, with a tooltip
/// saying how to let the level through, until a message of that level arrives
/// anyway, from a target with a directive of its own.
fn scope_bar(
    ui: &mut egui::Ui,
    picked: &mut [bool; Scope::ALL.len()],
    received: &[bool; Scope::ALL.len()],
    is_mac: bool,
) {
    let all = !picked.contains(&true);
    if ui.add(egui::Button::selectable(all, "All")).clicked() {
        *picked = Default::default();
    }
    let modifier = egui::ModifierNames::NAMES.format(&egui::Modifiers::COMMAND, is_mac);
    for scope in Scope::ALL {
        let index = scope as usize;
        let filtered_out = scope
            .level()
            .filter(|&level| !received[index] && !filter_default_allows(level));
        let mut response = ui
            .add_enabled(
                filtered_out.is_none(),
                egui::Button::selectable(picked[index], scope.label()),
            )
            .on_hover_text(format!("{modifier}-click to show several"));
        if let Some(level) = filtered_out {
            let level = level.as_str().to_lowercase();
            response = response.on_disabled_hover_text(format!(
                "LogPlugin filters out {level} messages. \
                 Restart with RUST_LOG={level} to show them."
            ));
        }
        if response.clicked() {
            if ui.input(|input| input.modifiers.command) {
                picked[index] = !picked[index];
            } else {
                *picked = Default::default();
                picked[index] = true;
            }
        }
    }
}

/// The command line: Enter submits it, ↑ and ↓ recall earlier lines.
fn prompt(ui: &mut egui::Ui, console: &mut Console) {
    let id = egui::Id::new("devtools_console_input");
    if ui.memory(|memory| memory.has_focus(id)) {
        // Consumed before the `TextEdit` sees them, which would otherwise move
        // the text cursor on ↑ and ↓.
        let (enter, up, down) = ui.input_mut(|input| {
            (
                input.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
            )
        });
        if enter {
            console.submit();
        }
        if up || down {
            console.prompt.browse(up);
            if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), id) {
                let end = egui::text::CCursor::new(console.prompt.line.chars().count());
                state
                    .cursor
                    .set_char_range(Some(egui::text::CCursorRange::one(end)));
                state.store(ui.ctx(), id);
            }
        }
    }
    ui.horizontal(|ui| {
        let line_height = ui.text_style_height(&egui::TextStyle::Monospace);
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ICON_WIDTH, line_height), egui::Sense::hover());
        let accent = ui.visuals().selection.stroke.color;
        paint_chevron(ui.painter(), rect.center(), accent);
        egui::TextEdit::singleline(&mut console.prompt.line)
            .id(id)
            .hint_text("help lists the commands")
            .frame(egui::Frame::NONE)
            .font(egui::TextStyle::Monospace)
            .desired_width(f32::INFINITY)
            // Enter is handled above, and must not end editing.
            .return_key(None)
            .show(ui);
    });
}

/// The rows the scope bar and the filter field let through. Only the rows in
/// view are laid out: the ones above are skipped by adding up their heights
/// from the last time they were drawn, with a one-line estimate for rows
/// never drawn.
fn message_list(ui: &mut egui::Ui, console: &mut Console, startup: Instant) {
    let filter = console.filter.trim();
    let shown: Vec<usize> = (0..console.rows.len())
        .filter(|&index| console.shows(&console.rows[index].message, filter))
        .collect();
    let estimate = ui.text_style_height(&egui::TextStyle::Body) + ROW_MARGIN.sum().y;
    let height = |row: &Row| {
        if row.height > 0.0 {
            row.height
        } else {
            estimate
        }
    };
    let total: f32 = shown
        .iter()
        .map(|&index| height(&console.rows[index]))
        .sum();

    let mut scroll_area = egui::ScrollArea::vertical()
        .auto_shrink(false)
        .stick_to_bottom(true);
    if std::mem::take(&mut console.scroll_to_bottom) {
        // Clamped to the end of the list.
        scroll_area = scroll_area.vertical_scroll_offset(total);
    }
    scroll_area.show_viewport(ui, |ui, viewport| {
        ui.set_height(total);
        // The viewport can lie past the end when the list just got shorter.
        let top = viewport.min.y.min(total - viewport.height()).max(0.0);
        let bottom = top + viewport.height();

        let mut y = 0.0;
        let mut first = shown.len();
        for (i, &index) in shown.iter().enumerate() {
            let row_height = height(&console.rows[index]);
            if y + row_height > top {
                first = i;
                break;
            }
            y += row_height;
        }

        let content = ui.max_rect();
        let rect = egui::Rect::from_x_y_ranges(
            content.x_range(),
            content.top() + y..=content.bottom().max(content.top() + y),
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
            let spacing = ui.spacing().item_spacing;
            // The rows touch; lines separate them.
            ui.spacing_mut().item_spacing.y = 0.0;
            for &index in &shown[first..] {
                if y >= bottom {
                    break;
                }
                let row = &mut console.rows[index];
                let rect = ui
                    .push_id(row.serial, |ui| message_row(ui, row, spacing, startup))
                    .inner
                    .rect;
                row.height = rect.height();
                y += rect.height();
            }
        });
    });
}

/// A row as in Safari: an icon for the kind of message, the repeat count, a
/// triangle expanding the details, the message with its fields, and the
/// target it was logged to on the right.
fn message_row(
    ui: &mut egui::Ui,
    row: &mut Row,
    spacing: egui::Vec2,
    startup: Instant,
) -> egui::Response {
    let message = &row.message;
    let visuals = ui.visuals();
    let fill = match message.kind {
        MessageKind::Log(Level::ERROR) | MessageKind::CommandError => {
            visuals.error_fg_color.gamma_multiply(0.15)
        }
        MessageKind::Log(Level::WARN) => visuals.warn_fg_color.gamma_multiply(0.15),
        _ => egui::Color32::TRANSPARENT,
    };
    // Commands and their output in monospace, so tables like clap's help line
    // up.
    let font = match message.source {
        Some(_) => egui::TextStyle::Body,
        None => egui::TextStyle::Monospace,
    };
    let line_height = ui.text_style_height(&font);
    let response = egui::Frame::new()
        .fill(fill)
        .inner_margin(ROW_MARGIN)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = spacing;
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let (icon, _) = ui
                    .allocate_exact_size(egui::vec2(ICON_WIDTH, line_height), egui::Sense::hover());
                paint_icon(ui.painter(), icon.center(), message.kind, ui.visuals());
                if row.count > 1 {
                    count_badge(ui, row.count, message.kind, line_height);
                }
                if message.source.is_some() {
                    disclosure_triangle(ui, &mut row.expanded, line_height);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    if let Some(source) = &message.source {
                        ui.weak(&source.target);
                    }
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        let summary = summary(ui, message, &font.resolve(ui.style()));
                        ui.add(egui::Label::new(summary).wrap());
                        if row.expanded {
                            details(ui, message, startup);
                        }
                    });
                });
            });
        })
        .response;
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    ui.painter()
        .hline(response.rect.x_range(), response.rect.bottom(), stroke);
    response
}

/// The message followed by its fields as `name=value`, like the log lines on
/// stderr.
fn summary(ui: &egui::Ui, message: &ConsoleMessage, font: &egui::FontId) -> egui::text::LayoutJob {
    let visuals = ui.visuals();
    let color = match message.kind {
        MessageKind::Log(Level::ERROR) | MessageKind::CommandError => visuals.error_fg_color,
        MessageKind::Log(Level::WARN) => visuals.warn_fg_color,
        MessageKind::Log(Level::TRACE) => visuals.weak_text_color(),
        MessageKind::Command => visuals.strong_text_color(),
        _ => visuals.text_color(),
    };
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &message.text,
        0.0,
        egui::TextFormat::simple(font.clone(), color),
    );
    for field in &message.fields {
        let space = if job.text.is_empty() { 0.0 } else { 8.0 };
        job.append(
            &format!("{}=", field.name),
            space,
            egui::TextFormat::simple(font.clone(), visuals.weak_text_color()),
        );
        append_value(&mut job, field, font, visuals);
    }
    job
}

/// The fields one per line, the spans the message was logged in, where it
/// was logged, and when.
fn details(ui: &mut egui::Ui, message: &ConsoleMessage, startup: Instant) {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let visuals = ui.visuals();
    let weak = egui::TextFormat::simple(font.clone(), visuals.weak_text_color());
    let normal = egui::TextFormat::simple(font.clone(), visuals.text_color());
    let line = |name: &str| {
        let mut job = egui::text::LayoutJob::default();
        job.append(&format!("{name}: "), 0.0, weak.clone());
        job
    };

    let mut lines = Vec::new();
    for field in &message.fields {
        let mut job = line(field.name);
        append_value(&mut job, field, &font, visuals);
        lines.push(job);
    }
    for span in &message.spans {
        let mut job = line("span");
        job.append(span.name, 0.0, normal.clone());
        if !span.fields.is_empty() {
            job.append("{", 0.0, weak.clone());
            for (i, field) in span.fields.iter().enumerate() {
                let space = if i == 0 { 0.0 } else { 6.0 };
                job.append(&format!("{}=", field.name), space, weak.clone());
                append_value(&mut job, field, &font, visuals);
            }
            job.append("}", 0.0, weak.clone());
        }
        lines.push(job);
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
        let mut job = line(name);
        job.append(&value, 0.0, normal.clone());
        lines.push(job);
    }

    for job in lines {
        ui.add(egui::Label::new(job).wrap());
    }
}

/// Appends a field's value, colored by type as in Xcode: strings red and
/// quoted, numbers yellow (blue in light mode), booleans pink, errors in the
/// error color.
fn append_value(
    job: &mut egui::text::LayoutJob,
    field: &Field,
    font: &egui::FontId,
    visuals: &egui::Visuals,
) {
    let dark = visuals.dark_mode;
    let rgb = |dark_rgb: [u8; 3], light_rgb: [u8; 3]| {
        let [r, g, b] = if dark { dark_rgb } else { light_rgb };
        egui::Color32::from_rgb(r, g, b)
    };
    let (text, color) = match field.kind {
        ValueKind::Str => (
            format!("{:?}", field.value),
            rgb([0xfc, 0x6a, 0x5d], [0xc4, 0x1a, 0x16]),
        ),
        ValueKind::Number => (
            field.value.clone(),
            rgb([0xd0, 0xbf, 0x69], [0x1c, 0x00, 0xcf]),
        ),
        ValueKind::Bool => (
            field.value.clone(),
            rgb([0xfc, 0x5f, 0xa3], [0x9b, 0x23, 0x93]),
        ),
        ValueKind::Error => (field.value.clone(), visuals.error_fg_color),
        ValueKind::Debug => (field.value.clone(), visuals.text_color()),
    };
    job.append(&text, 0.0, egui::TextFormat::simple(font.clone(), color));
}

/// Safari's repeat count: a pill with the number of times the message
/// arrived in a row.
fn count_badge(ui: &mut egui::Ui, count: u32, kind: MessageKind, line_height: f32) {
    let visuals = ui.visuals();
    let (fill, text_color) = match kind {
        MessageKind::Log(Level::ERROR) | MessageKind::CommandError => {
            (visuals.error_fg_color, egui::Color32::WHITE)
        }
        MessageKind::Log(Level::WARN) => (visuals.warn_fg_color, egui::Color32::BLACK),
        _ => (INFO_COLOR, egui::Color32::WHITE),
    };
    let galley = ui.painter().layout_no_wrap(
        count.to_string(),
        egui::FontId::proportional(10.0),
        text_color,
    );
    let pill_height = 14.0;
    let size = egui::vec2((galley.size().x + 8.0).max(pill_height), line_height);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let pill = egui::Rect::from_center_size(rect.center(), egui::vec2(size.x, pill_height));
    ui.painter()
        .rect_filled(pill, egui::CornerRadius::same(7), fill);
    ui.painter()
        .galley(pill.center() - galley.size() / 2.0, galley, text_color);
}

/// ▶ when collapsed, ▼ when expanded; a click toggles `expanded`.
fn disclosure_triangle(ui: &mut egui::Ui, expanded: &mut bool, line_height: f32) {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(10.0, line_height), egui::Sense::click());
    if response.clicked() {
        *expanded = !*expanded;
    }
    let color = if response.hovered() {
        ui.visuals().strong_text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    let c = rect.center();
    let points = if *expanded {
        vec![
            c + egui::vec2(-4.0, -2.0),
            c + egui::vec2(4.0, -2.0),
            c + egui::vec2(0.0, 3.0),
        ]
    } else {
        vec![
            c + egui::vec2(-2.0, -4.0),
            c + egui::vec2(3.0, 0.0),
            c + egui::vec2(-2.0, 4.0),
        ]
    };
    ui.painter().add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::NONE,
    ));
}

/// The icons follow Safari's: a red circle with "!" for errors, a yellow
/// triangle with "!" for warnings, a blue circle with "i" for infos, ">" for
/// a submitted line and "<·" for what it printed.
fn paint_icon(painter: &egui::Painter, c: egui::Pos2, kind: MessageKind, visuals: &egui::Visuals) {
    let v = egui::vec2;
    match kind {
        MessageKind::Log(Level::ERROR) | MessageKind::CommandError => {
            painter.circle_filled(c, 6.0, visuals.error_fg_color);
            paint_exclamation_mark(painter, c, egui::Color32::WHITE);
        }
        MessageKind::Log(Level::WARN) => {
            let points = vec![c + v(0.0, -6.5), c + v(6.5, 5.0), c + v(-6.5, 5.0)];
            painter.add(egui::Shape::convex_polygon(
                points,
                visuals.warn_fg_color,
                egui::Stroke::NONE,
            ));
            paint_exclamation_mark(painter, c + v(0.0, 0.5), egui::Color32::BLACK);
        }
        MessageKind::Log(Level::INFO) => {
            painter.circle_filled(c, 6.0, INFO_COLOR);
            painter.circle_filled(c + v(0.0, -3.0), 1.0, egui::Color32::WHITE);
            painter.line_segment(
                [c + v(0.0, -1.0), c + v(0.0, 3.5)],
                egui::Stroke::new(1.5, egui::Color32::WHITE),
            );
        }
        MessageKind::Log(Level::DEBUG) => {
            // Safari's icon for `console.log`: lines of text.
            let stroke = egui::Stroke::new(1.0, visuals.text_color());
            for dy in [-3.5, 0.0, 3.5] {
                painter.line_segment([c + v(-4.0, dy), c + v(4.0, dy)], stroke);
            }
        }
        MessageKind::Log(_) => {
            painter.circle_filled(c, 1.5, visuals.weak_text_color());
        }
        MessageKind::Command => paint_chevron(painter, c, visuals.selection.stroke.color),
        MessageKind::Output => {
            let stroke = egui::Stroke::new(1.5, visuals.weak_text_color());
            let tip = c + v(-4.0, 0.0);
            painter.line_segment([tip, tip + v(4.0, -4.0)], stroke);
            painter.line_segment([tip, tip + v(4.0, 4.0)], stroke);
            painter.circle_filled(c + v(3.5, 0.0), 1.2, visuals.weak_text_color());
        }
    }
}

fn paint_exclamation_mark(painter: &egui::Painter, c: egui::Pos2, color: egui::Color32) {
    painter.line_segment(
        [c + egui::vec2(0.0, -3.5), c + egui::vec2(0.0, 1.0)],
        egui::Stroke::new(1.5, color),
    );
    painter.circle_filled(c + egui::vec2(0.0, 3.2), 0.9, color);
}

/// ">", the prompt.
fn paint_chevron(painter: &egui::Painter, c: egui::Pos2, color: egui::Color32) {
    let stroke = egui::Stroke::new(1.5, color);
    let tip = c + egui::vec2(2.5, 0.0);
    painter.line_segment([tip + egui::vec2(-4.0, -4.0), tip], stroke);
    painter.line_segment([tip + egui::vec2(-4.0, 4.0), tip], stroke);
}

/// A trash can: lid, handle, and a body narrowing toward the bottom.
fn paint_trash(painter: &egui::Painter, c: egui::Pos2, stroke: egui::Stroke) {
    let v = egui::vec2;
    painter.line_segment([c + v(-5.0, -3.5), c + v(5.0, -3.5)], stroke);
    painter.line_segment([c + v(-1.5, -5.5), c + v(1.5, -5.5)], stroke);
    painter.add(egui::Shape::line(
        vec![
            c + v(-3.5, -3.5),
            c + v(-2.8, 5.0),
            c + v(2.8, 5.0),
            c + v(3.5, -3.5),
        ],
        stroke,
    ));
}
