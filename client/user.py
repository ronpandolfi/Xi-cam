from requests import Session

class User(object):
    """
    class for user credentials and sending and posting requests

    Attributes
    ----------
    session : requests.Session
    username : str
    logged_in : bool
        Boolean standing for login state. True if logged in
    """

    def __init__(self):
        super(User, self).__init__()
        self.session = Session()
        self.logged_in = False
        self.username = None

    def __del__(self):
        try:
            self.session.close()
        except TypeError:
            pass

    def login(self, username):
        """
        Sets the attributes according to login
        """
        self.username = username
        self.logged_in = True
        return self

    def logout(self):
        """
        When logging out
        """
        self.logged_in = False

    def check_login(self):
        """
        Raise an error if user is not logged in
        """
        if self.logged_in is False:
            raise AUTHError('%s is not logged in.' % self.username)

    def post(self, url, **kwargs):
        """
        Wrap session post
        """
        response = self.session.post(url, **kwargs)
        return response

    def get(self, url, **kwargs):
        """
         Wrap session get
         """
        response = self.session.get(url, **kwargs)
        return response

    @staticmethod
    def check_response(response):
        """
        Check for errors in a REST call
        """
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()

class AUTHError(Exception):
    """Error when users not logged in"""
    pass