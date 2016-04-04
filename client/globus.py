import base64
import json
import os
from urllib import quote
import time

from client.user import User


class GlobusClient(User):
    """
    Client class to authenticate and use globus online API for dataset
    transfering, moving, deleting, etc.
    """

    AUTH_URL = ("https://nexus.api.globusonline.org/goauth/token?grant_type"
                "=client_credentials")
    TRANSFER_URL = "https://transfer.api.globusonline.org/v0.10"
    STANDARD_ENDPOINTS = ('nersc#edison', 'nersc#cori', 'alsuser#bl832data')
    delete_document = os.path.abspath('./json/globusdelete.json')
    transfer_document = os.path.abspath('./json/globustransfer.json')

    def __init__(self):
        super(GlobusClient, self).__init__()
        self.endpoints = {}
        self.authentication = None

    def login(self, username, password):
        basic_auth = base64.b64encode('%s:%s'
                                      % (username, password))
        headers = {'Authorization': 'Basic %s' % basic_auth}
        r = self.get(self.AUTH_URL, headers=headers)

        if r.ok:
            access_token = r.json()['access_token']
            self.authentication = {'Authorization': 'Globus-Goauthtoken %s'
                                   % access_token}
            super(GlobusClient, self).login(username)
        else:
            self.authentication = None

    def add_standard_endpoints(self):
        """
        Add standard endpoints to the endpoint list
        """
        for endpoint in self.STANDARD_ENDPOINTS:
                self.add_endpoint(endpoint)

    def add_endpoint(self, endpoint):
        """
        Append globus endpoint as a key value pair
        where the key is the endpoint name and the value is the response
        in json format

        :param endpoint: str, endpoint name
        """
        self.endpoints[endpoint] = self.get_endpoint(endpoint)

    def get_endpoint(self, endpoint):
        """
        Get globus endpoint information as json

        :param endpoint: str, endpoint name
        """
        self.check_login()
        r = self.get(self.TRANSFER_URL + '/endpoint/' + quote(endpoint), headers=self.authentication)

        return self.check_response(r)

    def rm_endpoint(self, endpoint):
        """
        Remove an endpoint from the endpoint list

        :param endpoint: str, endpoint name
        """

        endpoint = self.endpoints.pop(endpoint)
        del endpoint

    def find_user_endpoints(self):
        """
        Find endpoints owned by user whose canonical name starts with the
        username as in user#*

        :return list of endpoint names
        """
        self.check_login()
        user_endpoints = []
        r = self.get(self.TRANSFER_URL + '/endpoint_list', headers=self.authentication)

        endpoints = self.check_response(r)
        for endpoint in endpoints['DATA']:
            if endpoint['username'] == self.username:
                user_endpoints.append(endpoint['canonical_name'])

        return user_endpoints

    def rm_endpoint(self, endpoint):
        """
        Remove and endpoint key value pair from endpoint dictionary instance
        variable

        :param endpoint: str, endpoint name
        """
        self.endpoints.pop(endpoint)

    def is_endpoint_active(self, endpoint):
        """
        Check if an endpoint is active

        :param endpoint: str, endpoint name
        :return: bool, true if active
        """
        self.check_login()
        self.add_endpoint(endpoint)
        if self.endpoints[endpoint]['activated'] is True:
            return True
        else:
            return False

    def determine_local_endpoint(self):
        """
        Determine which of the user's personal endpoints is the local machine
        if any. The function will essentially 'touch' a test file and then try
        to look for it through globus to find a match.

        :return: str, name of local endpoint, None if not found
        """
        self.check_login()
        user_endpoints = self.find_user_endpoints()
        try:
            home = os.environ['HOME']
        except KeyError:
            home = os.environ['HOMEPATH']

        test = os.path.join(home, 'test.spew')
        with open(test, 'wb'):
            os.utime(test, None)

        for endpoint in user_endpoints:
                params = {'path': unicode(test)}
                r = self.get(self.TRANSFER_URL + '/endpoint/' + quote(endpoint) + '/ls',
                             headers=self.authentication, params=params)
                try:
                    file_response = r.json()
                    if 'is a file' in file_response['message']:
                        os.remove(test)
                        return endpoint
                except KeyError:
                    pass

        os.remove(test)
        return None

    def get_dir_contents(self, path, endpoint):
        """
        Get contents of a specified directory in a given endpoint
        :param endpoint: str, endpoint name
        :param path: str, directory path
        :return: dict, dictionary with contents of directory
        """

        self.check_login()
        self.add_endpoint(endpoint)

        if self.endpoints[endpoint]['activated'] is True:
            params = {'path': path}
            r = self.get(self.TRANSFER_URL + '/endpoint/' + quote(endpoint) + '/ls',
                         headers=self.authentication, params=params)
            directory_items = self.check_response(r)
            directory_items = directory_items['DATA']
            return directory_items
        else:
            raise GLOBUSError('%s could not be activated' % endpoint)

    def transfer_file(self, src_endpoint, src_path, dst_enpoint, dst_path):
        """
        Make a file transfer submission to globus

        :param src_endpoint: str, source endpoint name
        :param src_path: str, path to file in source endpoint
        :param dst_enpoint: str, destination endpoint name
        :param dst_path: str, path to directory in destination endpoint including file name
        :return: json dict, globus response
        """
        self.check_login()

        for endpoint in (src_endpoint, dst_enpoint):
            if not self.is_endpoint_active(endpoint):
                raise GLOBUSError('%s endpoint is not active' % endpoint)

        r = self.get(self.TRANSFER_URL + '/submission_id',
                     headers=self.authentication)
        submission_id = self.check_response(r)

        with open(self.transfer_document) as json_file:
            transfer_submission = json.load(json_file)

        transfer_submission["submission_id"] = submission_id["value"]
        transfer_submission["source_endpoint"] = unicode(src_endpoint)
        transfer_submission["destination_endpoint"] = unicode(dst_enpoint)
        transfer_submission["DATA"][0]["source_path"] = unicode(src_path)
        transfer_submission["DATA"][0]["destination_path"] = unicode(dst_path)
        transfer_submission["label"] = unicode('transfered from SPEW client')

        r = self.post(self.TRANSFER_URL + '/transfer', json=transfer_submission, headers=self.authentication)
        transfer_result = self.check_response(r)

        return transfer_result

    def delete_file(self, endpoint, fpath):
        """
        Make a delete file submission to globus

        :param endpoint: str, endpoint name
        :param fpath: str, path to file to delete
        :return: json dict, globus response
        """

        self.check_login()

        if not self.is_endpoint_active(endpoint):
            raise GLOBUSError('%s endpoint is not active' % endpoint)

        r = self.get(self.TRANSFER_URL + '/submission_id', headers=self.authentication)
        submission_id = self.check_response(r)

        with open(self.delete_document) as json_file:
            delete_submission = json.load(json_file)

        delete_submission["submission_id"] = submission_id["value"]
        delete_submission["endpoint"] = unicode(endpoint)
        delete_submission["DATA"][0]["path"] = unicode(fpath)
        delete_submission["label"] = unicode('deleted from SPEW client')

        r = self.post(self.TRANSFER_URL + '/delete', json=delete_submission,
                      headers=self.authentication)
        delete_result = self.check_response(r)

        return delete_result


    def get_file_size(self, path, endpoint):
        """
        Get the size of a file in specified endpoint

        :param path: str, path to file
        :param endpoint: str, endpoint name
        :return: int, file size in bytes
        """

        self.check_login()

        path, name = os.path.split(path)
        r = self.get_dir_contents(path, endpoint)
        f = filter(lambda i: i['name'] == name, r)[0]

        return f['size']


    def get_task_status(self, task_id):
        """
        Make a GET call to monitor status of a submitted task

        :param id: str, taks id returned from a job submission
        :return:
        """
        self.check_login()

        r = self.get(self.TRANSFER_URL + '/task/' + task_id, headers=self.authentication)

        status = self.check_response(r)
        return status


    def transfer_generator(self, src_endpoint, src_path, dst_enpoint, dst_path):
        """
        Sumbits a globus transfer task and yields the fraction downloaded

        :param src_endpoint: str, source endpoint name
        :param src_path: str, path to file in source endpoint
        :param dst_enpoint: str, destination endpoint name
        :param dst_path: str, path to directory in destination endpoint including file name
        :return: json dict, globus response
        """

        self.check_login()

        size = self.get_file_size(src_path, src_endpoint)
        r = self.transfer_file(src_endpoint, src_path, dst_enpoint, dst_path)
        status = self.get_task_status(r['task_id'])
        while status['status'] == 'ACTIVE':
            time.sleep(3)
            status = self.get_task_status(r['task_id'])
            yield float(status['bytes_transferred'])/float(size)


class GLOBUSError(Exception):
    pass
