//! The "Graphics Debug" DevTools tab: a sidebar of pages on the left and the
//! selected page on the right. The sidebar groups the pages into sections
//! split by separators: information pages such as "GPU Info" first, then the
//! render stages, in the order they run within a frame, each showing its
//! debug toggles.

mod gpu;
mod screen_trace;
mod sharc;
mod sky;

use gpui::{
    AnyView, AppContext as _, Context, InteractiveElement as _, IntoElement, ParentElement as _,
    Render, StatefulInteractiveElement as _, Styled as _, Window, div, prelude::FluentBuilder as _,
    px,
};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::{ActiveTheme as _, h_flex, v_flex};

struct Page {
    name: &'static str,
    /// The page's view; `None` for a render stage without debug toggles.
    contents: Option<AnyView>,
}

pub(crate) struct GraphicsDebugView {
    /// The sidebar's sections, top to bottom.
    sections: Vec<Vec<Page>>,
    /// Section and page index of the shown page.
    selected: (usize, usize),
}

impl GraphicsDebugView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let page = |name, contents: Option<AnyView>| Page { name, contents };
        let info = vec![page(
            "GPU Info",
            Some(cx.new(|cx| gpu::GpuInfoPage::new(window, cx)).into()),
        )];
        let render_pipeline = vec![
            page(
                "Sky atmosphere",
                Some(cx.new(|cx| sky::SkyPage::new(window, cx)).into()),
            ),
            page("Primary rays", None),
            page("HZB build", None),
            page("Shadow rays", None),
            page(
                "Screen trace",
                Some(
                    cx.new(|cx| screen_trace::ScreenTracePage::new(window, cx))
                        .into(),
                ),
            ),
            page("Final gather", None),
            page(
                "SHARC",
                Some(cx.new(|cx| sharc::SharcPage::new(window, cx)).into()),
            ),
            page("Auto-exposure", None),
            page("Super-resolution", None),
            page("Tonemap", None),
        ];
        GraphicsDebugView {
            sections: vec![info, render_pipeline],
            // "GPU Info".
            selected: (0, 0),
        }
    }
}

impl Render for GraphicsDebugView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let mut sidebar = v_flex()
            .id("graphics-debug-pages")
            .w(px(160.))
            .flex_shrink_0()
            .h_full()
            .p_1()
            .border_r_1()
            .border_color(theme.border)
            .overflow_y_scroll();
        for (s, pages) in self.sections.iter().enumerate() {
            if s > 0 {
                sidebar = sidebar.child(div().my_1().h(px(1.)).bg(theme.border));
            }
            for (p, page) in pages.iter().enumerate() {
                let selected = self.selected == (s, p);
                sidebar = sidebar.child(
                    div()
                        .id(("graphics-debug-page", s * 100 + p))
                        .px_2()
                        .py_0p5()
                        .rounded_sm()
                        .when(page.contents.is_none(), |item| {
                            item.text_color(theme.muted_foreground)
                        })
                        .when(selected, |item| item.bg(theme.list_active))
                        .hover(|item| item.bg(theme.list_hover))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.selected = (s, p);
                            cx.notify();
                        }))
                        .child(page.name),
                );
            }
        }
        let (s, p) = self.selected;
        let contents = match self.sections[s][p].contents.clone() {
            Some(view) => div().child(view),
            None => div()
                .text_color(theme.muted_foreground)
                .child("No debug toggles for this stage."),
        };
        h_flex().size_full().items_start().child(sidebar).child(
            div()
                .flex_1()
                .h_full()
                .p_2()
                .overflow_y_scrollbar()
                .child(contents),
        )
    }
}
