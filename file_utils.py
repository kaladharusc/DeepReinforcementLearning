import json

def write_progress(progress_data):
    f = open("data/progress.txt", "w+")
    f.write(json.dumps(progress_data))
    f.close()

def write_output(output_data):
    f = open("./communicate/output.txt", "w")
    f.write(output_data)
    f.close()