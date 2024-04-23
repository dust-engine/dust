use rhyolite_rtx::SbtManager;

pub struct PbrPipeline {
    manager: SbtManager<MaterialSbtMarker<PbrPipeline>, 2>,
}
