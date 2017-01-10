import os
from requests import HTTPError
from client.user import User
from pipeline import msg

class NewtClient(User):
    """
    Client class to handle Newt API calls

    Attributes
    ----------
    authentication : str
        authentication token
    scratch_dir : str
        path to a user's scratch directory on NERSC system
    home_dir : str
        path to a user's scratch directory on NERSC system
    """

    BASE_URL = "https://newt.nersc.gov/newt"
    systems = ('edison', 'cori')

    def __init__(self):
        super(NewtClient, self).__init__()
        self.authentication = None  # NewtClient.login(self)
        self.scratch_dir = None
        self.home_dir = None

    def __del__(self):
        if self.logged_in:
            self.logout()
        try:
            self.session.close()
        except TypeError:
            pass

    def login(self, username, password):
        """
        Login to NEWT

        Parameters
        ----------
        username : str
        password : str
        """

        credentials = {"username": username,
                       "password": password}

        response = self.post(self.BASE_URL + "/auth", data=credentials)

        if response.json()['auth']:
            self.authentication = response
            return super(NewtClient, self).login(username)
        else:
            self.authentication = None
            raise NEWTError('Bad Authentication: Unable to log in')

    def logout(self):
        """
        Send REST call to logout from NEWT
        """
        self.authentication = self.post(self.BASE_URL + '/logout')
        User.logout(self)

    @classmethod
    def check_system(cls, system):
        """
        Check if the given system is in the class's allowed systems list.
        Raise an error if not

        Parameters
        ----------
        system : str
            NERSC system name
        """

        if system not in cls.systems:
            raise NEWTError('%s is not a NERSC system' % system)

    def get_system_status(self, system='all'):
        """
        Get system status via NEWT API status call

        Parameters
        ----------
        system : str
            NERSC system name ('all' for all systems)

        Returns
        -------
        json
            status response
        """

        self.check_login()
        if system is 'all':
            system = ''
        else:
            self.check_system(system)

        r = self.post(self.BASE_URL + '/status/' + system)
        return self.check_response(r)

    def get_dir_contents(self, path, system):
        """
        Get contents of a directory on a NERSC system

        Parameters
        ----------
        path : str
            path of directory
        system : str
            NERSC system name

        Returns
        -------
        json
            response of directory contents
        """

        self.check_login()
        self.check_system(system)

        r = self.get(self.BASE_URL + '/file/' + system + path)

        return self.check_response(r)

    def get_file_size(self, path, system):
        """
        Get the size of a file in bytes

        Parameters
        ----------
        path : str
            path to file

        Returns
        -------
        int
            file size
        """

        info = self.get_dir_contents(path, system)
        if len(info) > 1:
            raise NEWTError('{} is a directory'.format(path))

        return int(info[0]['size'])

    def get_scratch_dir(self, system):
        """
        Get a users scratch directory on a NERSC platform

        Parameters
        ----------
        system : str, optional
            NERSC system name

        Returns
        -------
        str
            path of user's scratch if found. None, if not found
        """

        self.check_login()
        scratch_dir = None
        if system == 'edison':
            for i in range(1, 10):
                root_folder = ('/scratch' + str(i) + '/scratchdirs/' +
                               self.username)
                if self._check_scratch_path(root_folder, system):
                    scratch_dir = root_folder
        elif system == 'cori':
            root_folder = '/global/cscratch1/sd/' + self.username
            if self._check_scratch_path(root_folder, system):
                scratch_dir = root_folder

        if scratch_dir is None:
            # raise NEWTError('User %s scratch directory not found'
            #                  % self.username)
            return '/'
        return scratch_dir

    def set_scratch_dir(self, system):
        """
        Get a users scratch directory on a NERSC platform and set it to the instance variable

        Parameters
        ----------
        system : str, optional
            NERSC system name
        """
        self.scratch_dir = self.get_scratch_dir(system)

    def _check_scratch_path(self, path, system):
        """
        Checks to see if the path is accessible by NEWT

        Parameters
        ----------
        path : str
        system : str
            NERSC system
        """

        try:
            contents = self.get_dir_contents(path, system)
            for item in contents:
                if 'user' in item.keys():
                    if item['user'] == self.username:
                        return True
        except HTTPError:
            return False

    def get_home_dir(self, system):
        """
        Get a user's home directory on a NERSC system

        Parameters
        ----------
        system : str
            NERSC system name

        Returns
        -------
        str
            path of users home directory if found. None, if not found
        """

        self.check_login()
        home_dir = None
        try:
            root_home = ('/global/homes/' + self.username[0] + '/' +
                         self.username)
            contents = self.get_dir_contents(root_home, system)
            for item in contents:
                    if 'user' in item.keys():
                        if item['user'] == self.username:
                            home_dir = root_home
                            break
        except:
            pass

        return home_dir

    def set_home_dir(self, system):
        """
        Get a users home directory on a NERSC platform and set it to the instance variable

        Parameters
        ----------
        system : str, optional
            NERSC system name
        """

        self.home_dir = self.get_home_dir(system)

    def execute_command(self, command, system, bin_path='/bin'):
        """
        Execute a system command on a NERSC system

        Parameters
        ----------
        command : str
            command including any arguments
        system : str
            NERSC system name
        bin_path : str
            path to binary
        """

        self.check_login()
        self.check_system(system)

        r = self.post(self.BASE_URL + '/command/' + system,
                      data={'executable': bin_path + '/' + command})

        return self.check_response(r)

    def delete_file(self, fpath, system):
        """
        Delete a specific file on a NERSC system, 'rm -rf' command

        Parameters
        ----------
        fpath : str
            path to file
        system : str
            NERSC system name
        """

        command = 'rm -rf ' + fpath
        return self.execute_command(command, system)

    def move_file(self, fpath, dst_path, system):
        """
        Move a specified file on a NERSC system. 'mv' command

        Parameters
        ----------
        fpath : str
            path to file
        dst_path : str
            path to new directory
        system: str
            NERSC system name
        """

        command = 'mv ' + fpath + ' ' + dst_path
        return self.execute_command(command, system)

    def copy_file(self, fpath, dst_path, system):
        """
        Copy a specified file to a new location on a NERSC system.
        'cp -ar' command

        Parameters
        ----------
        fpath : str
            path to file
        dst_path : str
            path to copy into
        system : str
            NERSC system name
        """
        command = 'cp -ar ' + fpath + ' ' + dst_path
        return self.execute_command(command, system)

    def rsync(self, fpath, dst_path, system, *args):
        """
        Use rsync locally on a nersc system

        Parameters
        ----------
        fpath : str
            source path
        dst_path : str
            dest path
        system : str
            nersc system
        args : str
            lags for rsync (ie '-p'
        """
        command = 'rsync '
        for arg in args:
            command += arg + ' '
        command += fpath + ' ' + dst_path

        return self.execute_command(command, system, bin_path='/usr/bin')

    def upload_file(self, fpath, dpath, system):
        """
        Upload a file (<100 MB or it will not work)

        Parameters
        ----------
        fpath : str
            Local path to file
        dpath : str
            Destination path
        system : str
            NERSC system
        """

        self.check_login()
        self.check_system(system)
        fname = os.path.split(fpath)[-1]
        with open(fpath, 'rb') as f:
            file = {'file': (fpath, f), 'file_name': fname}
            response = self.post(self.BASE_URL + '/file/' + system + dpath, files=file)

        return response

    def download_file(self, path, system, save_path=None):
        """
        Download a file on a NERSC system. (<100 MB or it will not work)

        Parameters
        ----------
        path : str
            file path on NERSC
        system : str
            NERSC system
        save_path : str
            path and name to save file locally. If None name on NERSC is used and save in home directory
        :return:
        """

        self.check_login()

        if save_path is None:
            save_path = os.path.join(os.path.expanduser('~'), os.path.split(path)[-1])

        r = self.get(self.BASE_URL + '/file/' + system + '/' +  path + '?view=read', stream=True)

        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=64*1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        r.close()

        return

    def download_file_generator(self, path, system, save_path=None, chunk_size=64*1024):
        """
        Download a dataset as a generator (yields the fraction downloaded)
        Useful to know the status of a download (for gui purposes)

        Parameters
        ----------
        path : str
            file path on NERSC
        system : str
            NERSC system
        save_path : str
            path and name to save file locally. If None name on NERSC is used and save in home directory
        chunk_size: in, optional
            Chunks size in bytes

        Yields
        ------
        float
            percent downloaded
        """
        self.check_login()

        if save_path is None:
            save_path = os.path.join(os.path.expanduser('~'), os.path.split(path)[-1])

        file_size = float(self.get_file_size(path, system))
        r = self.get(self.BASE_URL + '/file/' + system + '/' + path + '?view=read', stream=True)

        with open(save_path, 'wb') as f:
            downloaded = 0.0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
                    downloaded += len(chunk)/file_size
                    yield downloaded
        r.close()


class NEWTError(Exception):
    """Raised when a parameter that will be used for the NEWT API is invalid"""
    pass
