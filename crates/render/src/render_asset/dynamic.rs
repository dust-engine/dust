
/// Render asset that can be updated.
pub trait DynamicRenderAsset: RenderAsset {
    type GPUMaterial: GPUDynamicRenderAsset<Self>;

    type CreateUpdateDataParam: SystemParam;
    type UpdateData: Send + Sync;
    fn create_update_data(
        &mut self,
        param: &mut SystemParamItem<Self::CreateUpdateDataParam>,
    ) -> Self::UpdateData;
}

pub trait GPUDynamicRenderAsset<T: DynamicRenderAsset>: Send + Sync {
    type UpdateParam: SystemParam;
    fn update(
        &mut self,
        change_set: T::UpdateData,
        commands_future: &mut CommandsFuture,
        params: &mut SystemParamItem<Self::UpdateParam>,
    );
}

