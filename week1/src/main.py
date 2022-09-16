import requests
import json


REQUEST_URI = 'http://www.trans-gps.cv.ua/map/tracker/?selectedRoutesStr='


def request_data(req_uri):
    try:
        with requests.get(req_uri) as request:
            if (request is None) or (request.text is None):
                return
            return json.loads(request.text)
    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as err:
        print("Error while trying to GET data")
        print(err)

    return None


def main():
    print(f"requests.__version__={requests.__version__}")
    print(f"json.__version__={json.__version__}")
    response = request_data(REQUEST_URI)
    print(f"Response keys are {response.keys()}")

if __name__ == "__main__":
    main()