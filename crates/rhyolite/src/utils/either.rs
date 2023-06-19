use crate::future::RenderData;

pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A: RenderData, B: RenderData> RenderData for Either<A, B> {
    fn tracking_feedback(&mut self, feedback: &crate::future::TrackingFeedback) {
        match self {
            Either::Left(a) => a.tracking_feedback(feedback),
            Either::Right(a) => a.tracking_feedback(feedback),
        }
    }
}
