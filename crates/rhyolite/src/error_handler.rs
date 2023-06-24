use ash::vk;

static DEVICE_LOST_HANDLER: std::sync::OnceLock<fn ()> = std::sync::OnceLock::new();


pub fn handle_device_lost(error: vk::Result) -> vk::Result {
    if error != vk::Result::ERROR_DEVICE_LOST {
        return error;
    }
    if let Some(handler) = DEVICE_LOST_HANDLER.get() {
        (handler)();
    }
    panic!()
}

pub fn set_global_device_lost_handler(handler: fn()) -> Result<(), fn()> {
    DEVICE_LOST_HANDLER.set(handler)
}
