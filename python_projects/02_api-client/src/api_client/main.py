import requests

API_URL = "https://jsonplaceholder.typicode.com/posts/1"


def get_post():
    print("Calling API...")
    response = requests.get(API_URL)

    if response.status_code == 200:
        data = response.json()
        print("Post title:", data["title"])
        print("Post body:", data["body"])
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    # Run this code ONLY when the file is executed directly — NOT when imported.
    get_post()
