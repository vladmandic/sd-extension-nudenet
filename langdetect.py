repo_id = "facebook/fasttext-language-identification"
model = None


def lang_detect(text:str, top:int=1, threshold:float=0.25) -> str:
    try:
        global model # pylint: disable=global-statement
        from modules import shared
        if model is None:
            from installer import install
            install("fasttext")
            import fasttext
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id, filename="model.bin", cache_dir=shared.opts.hfcache_dir)
            shared.log.info(f'Load: model="{repo_id}"')
            model = fasttext.load_model(model_path)
        text = text.replace('\n', '. ')
        lang, score = model.predict(text, k=top, threshold=threshold, on_unicode_error="ignore")
        result = [f"{l.replace("__label__", "").lower()}:{s:.2f}" for l, s in zip(lang, score) if s > threshold][:top]
        shared.log.debug(f'LangDetect: {result}')
        return result
    except Exception as e:
        shared.log.error(f'LangDetect: {e}')
        return str(e)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <text>")
    else:
        print(lang_detect(sys.argv[1]))
