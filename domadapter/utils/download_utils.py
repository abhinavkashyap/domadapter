import requests


class GoogleDriveDownloader:
    def __init__(self):
        pass

    def download_file_from_google_drive(self, file_id: str, destination: str):
        """Downloads file from google drive.

        Parameters
        ----------
        file_id: str
            Right click on the file.
            Click on Share. In the "Get Link" section of the pop-up click on
            change. Copy the link and use the file id.
            https://drive.google.com/file/d/fileid/view?usp=sharing
        destination: str
            The destination filename

        Returns
        -------
        None

        """
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={"id": file_id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        self.save_response_content(response, destination)

    @staticmethod
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    @staticmethod
    def save_response_content(response, destination: str):
        """"""
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
