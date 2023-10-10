import gradio as gr
from modules import scripts, processing
import nudenet


class Script(scripts.Script):
    def title(self):
        return 'NudeNet'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img): # pylint: disable=unused-argument
        with gr.Accordion('NudeNet', open = False, elem_id='nudenet'):
            with gr.Row():
                enabled = gr.Checkbox(label = 'Enabled', value = False)
                threshold = gr.Number(label = 'Threshold', value = 0.2, min = 0, max = 1, step = 0.01, interactive=True)
                method = gr.Dropdown(label = 'Method', value = 'pixelate', choices = ['none', 'blur', 'pixelate', 'block'], interactive=True)
            with gr.Row():
                censor = gr.Dropdown(label = 'Censor', value = [], choices = nudenet.labels, multiselect=True, interactive=True)
        return [enabled, method, censor, threshold]

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, enabled, method, censor, threshold): # pylint: disable=arguments-differ
        if not enabled:
            return
        if nudenet.detector is None:
            nudenet.detector = nudenet.NudeDetector()
        nudes = nudenet.detector.censor(image=pp.image, method=method, min_score=threshold, censor=censor)
        if len(censor) > 0:
            pp.image = nudes.output
        p.extra_generation_params["NudeNet"] = '; '.join([f'{d["label"]}:{d["score"]}' for d in nudes.detections])
