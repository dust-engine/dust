import Cocoa
import Metal

class Renderer {
    var device: MTLDevice
    var metalLayer: CAMetalLayer

    init(view: NSView) {
        self.device = MTLCreateSystemDefaultDevice()!
        self.metalLayer = CAMetalLayer()
        self.metalLayer.device = self.device
        self.metalLayer.pixelFormat = .bgra8Unorm
        self.metalLayer.framebufferOnly = true
        self.metalLayer.frame = view.layer!.frame
        view.layer!.addSublayer(metalLayer)
    }
}

@_cdecl("RendererNew")
func RendererNew(
    window: NSWindow,
    view: NSView
) {
    let renderer = Renderer(view: view)
}
