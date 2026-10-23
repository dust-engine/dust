//! Controls shared by the panels.

use std::ops::RangeInclusive;

use gpui::{
    App, AppContext as _, Context, Entity, IntoElement, ParentElement as _, SharedString,
    Styled as _, Subscription, Window, div, px, relative,
};
use gpui_component::slider::{Slider, SliderEvent, SliderScale, SliderState};
use gpui_component::{ActiveTheme as _, h_flex};

/// A labeled slider for one number of the world.
///
/// Dragging it calls `on_change`, which sends the edit; [`Self::set`] shows a
/// value read from the world.
pub(crate) struct NumberSlider {
    label: SharedString,
    state: Entity<SliderState>,
    /// Decimals shown; 0 for an integer.
    decimals: usize,
    _subscription: Subscription,
}

impl NumberSlider {
    pub fn new<V: 'static>(
        label: impl Into<SharedString>,
        range: RangeInclusive<f32>,
        step: f32,
        logarithmic: bool,
        window: &mut Window,
        cx: &mut Context<V>,
        on_change: impl Fn(&mut V, f32, &mut Context<V>) + 'static,
    ) -> Self {
        let state = cx.new(|_| {
            // `max` first: `min` clamps against the default maximum of 100.
            SliderState::new()
                .max(*range.end())
                .min(*range.start())
                .step(step)
                .scale(if logarithmic {
                    SliderScale::Logarithmic
                } else {
                    SliderScale::Linear
                })
                .default_value(*range.start())
        });
        let subscription = cx.subscribe_in(&state, window, move |this, _, event, _, cx| {
            if let SliderEvent::Change(value) = event {
                on_change(this, value.end(), cx);
            }
        });
        let decimals = if step >= 1.0 {
            0
        } else {
            (-step.log10()).ceil() as usize
        };
        NumberSlider {
            label: label.into(),
            state,
            decimals,
            _subscription: subscription,
        }
    }

    /// Shows `value`, without calling `on_change`.
    pub fn set(&self, value: f32, window: &mut Window, cx: &mut App) {
        self.state.update(cx, |state, cx| {
            if state.value().end() != value {
                state.set_value(value, window, cx);
            }
        });
    }

    pub fn render(&self, cx: &App) -> impl IntoElement + use<> {
        let value = self.state.read(cx).value().end();
        // The label takes a share of the row rather than a fixed width, so the
        // slider keeps a usable length in a narrow dock.
        h_flex()
            .gap_2()
            .child(
                div()
                    .w(relative(0.4))
                    .max_w(px(180.))
                    .flex_shrink_0()
                    // A wrapped label's lines as close as a checkbox's.
                    .line_height(relative(1.25))
                    .text_color(cx.theme().muted_foreground)
                    .child(self.label.clone()),
            )
            .child(Slider::new(&self.state).flex_1().min_w(px(48.)))
            .child(
                div()
                    .w(px(52.))
                    .flex_shrink_0()
                    .text_right()
                    .font_family(cx.theme().mono_font_family.clone())
                    .child(format!("{value:.*}", self.decimals)),
            )
    }
}

/// Room on the right of a `RadioGroup` for its labels. gpui-component makes a
/// radio's label as wide as the whole radio, beside the indicator, so a label
/// that wraps can run past the group by the indicator's width and the gap.
pub(crate) const RADIO_LABEL_OVERHANG: f32 = 24.0;

/// A bold heading over a group of controls.
pub(crate) fn heading(text: impl Into<SharedString>) -> impl IntoElement {
    div()
        .mt_2()
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .child(text.into())
}
