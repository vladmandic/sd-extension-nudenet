import gradio as gr
from modules import scripts, processing # pylint: disable=import-error
import nudenet


class Script(scripts.Script):
    def title(self):
        return 'NudeNet'

    def show(self, is_img2img): # pylint: disable=unused-argument
        return scripts.AlwaysVisible

    def ui(self, is_img2img): # pylint: disable=unused-argument
        with gr.Accordion('NudeNet', open = False, elem_id='nudenet'):
            with gr.Row():
                enabled = gr.Checkbox(label = 'Enabled', value = False)
                min_score = gr.Number(label = 'Threshold', value = 0.2, min = 0, max = 1, step = 0.01, interactive=True)
                method = gr.Dropdown(label = 'Method', value = 'pixelate', choices = ['none', 'blur', 'pixelate', 'block'], interactive=True)
            with gr.Row():
                censor = gr.Dropdown(label = 'Censor', value = [], choices = nudenet.labels, multiselect=True, interactive=True)
        return [enabled, method, censor, min_score]

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, enabled, method, censor, min_score): # pylint: disable=arguments-differ
        if not enabled:
            return
        if nudenet.detector is None:
            nudenet.detector = nudenet.NudeDetector() # loads and initializes model once
        nudes = nudenet.detector.censor(image=pp.image, method=method, min_score=min_score, censor=censor)
        if len(censor) > 0: # replace image if anything is censored
            pp.image = nudes.output
        p.extra_generation_params["NudeNet"] = '; '.join([f'{d["label"]}:{d["score"]}' for d in nudes.detections]) # add all metadata
