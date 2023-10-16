import os

from core.utils import read_img, save_img, build_model
from core.infer import infer_single_image

from options.infer_options import InferOptions

if __name__ == "__main__":
    options = InferOptions()
    opts = options.parse_args()

    os.makedirs(opts.output_path, exist_ok=True)

    # Infer on single image
    img_name = opts.input_path.split(sep=os.sep)[-1].split(sep=".")[0]
    model = build_model(opts.config_path, opts.checkpoint_path)
    x = read_img(opts.input_path, 256)

    x_trg = infer_single_image(model, x, opts.mode, 
                               tag=opts.tag, attribute=opts.attribute, z=opts.z, 
                               ref_img_path=opts.reference_path, 
                               device=opts.device)
    if opts.mode == "l":
        x_trg = infer_single_image(model, x, opts.mode, 
                               tag=opts.tag, attribute=opts.attribute, z=opts.z, 
                               device=opts.device)
        out_file_name = f"{img_name}_out_tag:{opts.tag}_attr:{opts.attribute}_z:{opts.z}.jpg"
    else:
        x_trg = infer_single_image(model, x, opts.mode, 
                               tag=opts.tag, 
                               ref_img_path=opts.reference_path, 
                               device=opts.device)
        out_file_name = f"{img_name}_out_tag:{opts.tag}.jpg"
    save_img(x_trg, opts.output_path, out_file_name)
    print("Inference done!")

    