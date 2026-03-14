try:
    import model_loader
except Exception as e:
    import traceback
    with open("err.log", "w") as f:
        f.write(traceback.format_exc())
