use bevy::app::{App, Plugin};
use tracing_subscriber::{prelude::*, EnvFilter, Registry};

#[cfg(feature = "sentry")]
mod sentry;

pub struct LogPlugin;
impl Plugin for LogPlugin {
    fn build(&self, app: &mut App) {
        let subscriber = Registry::default();

        let fmt_filter_layer = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(&"info"))
            .unwrap();
        let fmt_layer = tracing_subscriber::fmt::Layer::default()
            .with_writer(std::io::stderr)
            .with_filter(fmt_filter_layer);

        let subscriber = subscriber.with(fmt_layer);

        #[cfg(feature = "sentry")]
        app.add_plugins(sentry::SentryPlugin);
        #[cfg(feature = "sentry")]
        let subscriber = sentry::update_subscriber(subscriber);

        subscriber.init();
    }
}
