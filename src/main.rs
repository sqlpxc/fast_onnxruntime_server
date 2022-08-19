use axum::body::Bytes;
use axum::response::Response;
use axum::{routing::post, Router};
use image::{self, EncodableLayout};
use once_cell::sync::OnceCell;
use std::net::SocketAddr;
use tract_ndarray::Array;
use tract_onnx::prelude::*;

static MODEL_IMAGE_CLASSIFICATION: OnceCell<ImageClassification> = OnceCell::new();

struct ImageClassification {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl ImageClassification {
    fn new(onnx_path: String) -> Self {
        let model = tract_onnx::onnx()
            .model_for_path(onnx_path)
            .unwrap()
            .with_input_fact(0, f32::fact(&[1, 3, 224, 224]).into())
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        ImageClassification { model: model }
    }

    fn run(&self, raw_img: Vec<u8>) -> i32 {
        let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.485, 0.456, 0.406]).unwrap();
        let std = Array::from_shape_vec((1, 3, 1, 1), vec![0.229, 0.224, 0.225]).unwrap();

        let img = image::load_from_memory(&raw_img).unwrap().to_rgb8();
        let resized =
            image::imageops::resize(&img, 224, 224, ::image::imageops::FilterType::Triangle);

        let image: Tensor =
            ((tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
                resized[(x as _, y as _)][c] as f32 / 255.0
            }) - mean)
                / std)
                .into();

        let result = self.model.run(tvec!(image)).unwrap();

        let best = result[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .zip(1..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("result: {:?}", best);
        best.unwrap().1
    }
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", post(root))
        .route("/startRuntime", post(start_runtime));

    let addr = SocketAddr::from(([127, 0, 0, 1], 6666));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn root(raw_img: Bytes) -> String {
    let index = MODEL_IMAGE_CLASSIFICATION
        .get()
        .unwrap()
        .run(raw_img.to_vec());
    index.to_string()
}

async fn start_runtime(onnx_path: Bytes) -> Response<String> {
    println!(
        "{}",
        String::from_utf8_lossy(onnx_path.as_bytes()).to_string()
    );
    MODEL_IMAGE_CLASSIFICATION.get_or_init(|| {
        ImageClassification::new(String::from_utf8_lossy(onnx_path.as_bytes()).to_string())
    });

    Response::builder()
        .status(200)
        .body(String::from("ok"))
        .unwrap()
}
