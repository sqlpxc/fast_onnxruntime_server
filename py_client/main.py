import urllib.request
import urllib.parse


def start_runtime():
    response = urllib.request.urlopen(
        "http://127.0.0.1:6666/startRuntime", data=bytes("resnetttt.onnx", encoding='utf-8'))
    print(response.read().decode('utf-8'))


def run():
    raw_img = open("b.jpg", "rb").read()
    response = urllib.request.urlopen("http://127.0.0.1:6666", data=raw_img)
    print(response.read().decode('utf-8'))


if __name__ == "__main__":
    start_runtime()
    for i in range(1, 100):
        run()
