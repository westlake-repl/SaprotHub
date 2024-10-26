import uvicorn
import requests

from fastapi import FastAPI


app = FastAPI()


@app.get("/fetch")
def fetch_page(url: str):
    """
    This function is used fetch web pages and modify the X-Frame-Options header to allow embedding in iframes
    Args:
        url: URL of the web page to fetch
    """
    response = requests.get(url)
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response.text


if __name__ == "__main__":
    # Launch
    uvicorn.run("iframe_proxy:app", host="0.0.0.0", port=8000)