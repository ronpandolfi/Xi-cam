from requests import Session

class User(object):
    """
    class for user credentials and sending and posting requests
    """

    def __init__(self):
        super(User, self).__init__()
        self.session = Session()
        self.logged_in = False
        self.username = None

    def __del__(self):
        # Something weird is going on here??
        try:
            self.session.close()
        except TypeError:
            pass

    def login(self, username):
        self.username = username
        self.logged_in = True

    def logout(self):
        self.logged_in = False

    def check_login(self):
        if self.logged_in is False:
            raise AUTHError('%s is not logged in.' % self.username)

    def post(self, url, **kwargs):
        response = self.session.post(url, **kwargs)
        return response

    def get(self, url, **kwargs):
        response = self.session.get(url, **kwargs)
        return response

    @staticmethod
    def check_response(response):
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()

class AUTHError(Exception):
    pass