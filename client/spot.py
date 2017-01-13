# -*- coding: utf-8 -*-
import os
from time import sleep
from StringIO import StringIO
from PIL import Image
import numpy as np
from client.newt import NewtClient


class SpotClient(NewtClient):
    """
    Client class to handle SPOT API calls
    """

    BASE_DIR = '/global/project/projectdirs/als/spade/warehouse'
    SPOT_URL = 'https://portal-auth.nersc.gov/als'

    def __init__(self):
        super(SpotClient, self).__init__()
        self.spot_authentication = None

    def login(self, username, password):
        """
        Login to SPOT

        Parameters
        ----------
        username : str
        password : str
        """

        credentials = {"username": username,
                       "password": password}
        response = self.post(self.SPOT_URL + '/auth', data=credentials)
        if response.json()['auth']:
            self.authentication = response
            return super(SpotClient, self).login(username, password)
        else:
            self.authentication = None
            raise SPOTError('Bad Authentication: Unable to log in')

    def search(self, query, **kwargs):
        """
        Search a dataset on SPOT

        Parameters
        ----------
        query : str, search query
        kwargs : {key: option}
            Any of the following:
            'sortterm'
                attribute to sort results by
            'sorttype'
                ascending 'asc' or descending 'desc'
            'end_station'
                endstation of dataset
            'limitnum'
                maximum number of results to show
            'skipnum'
                number of results to skipd
            'search'
                search query

        Returns
        -------
        json
            response of search results
        """
        self.check_login()
        # sortterm probably allows more filters will need to check...
        allowed_kwargs = {'sortterm': ['fs.stage_date', 'appmetadata.sdate'],
                          'sorttype': ['desc', 'asc'],
                          'end_station': ['bl832', 'bl733']}
        generic_kwargs = ['limitnum', 'skipnum']
        for key in kwargs:
            if key not in allowed_kwargs.keys() and key not in generic_kwargs:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, list(allowed_kwargs.keys() +
                                  generic_kwargs)))
            elif key in allowed_kwargs:
                if kwargs[key] not in allowed_kwargs[key]:
                    raise ValueError('%s keyword value must be on of %s' %
                                     (kwargs[key], list(allowed_kwargs[key])))
        kwargs.update(search=query)

        r = self.get(self.SPOT_URL + '/hdf/search', params=kwargs)

        return self.check_response(r)

    def get_derived_datasets(self, dataset):
        """
        Get datasets that are derived from the given dataset. ie raw, sino,
        norm, etc...

        Parameters
        ----------
        dataset : str
            dataset name

        Returns
        -------
        json
            Response with derived datasets
        """

        self.check_login()
        params = {'dataset': dataset}
        r = self.post(self.SPOT_URL + '/hdf/dataset', params=params)
        return self.check_response(r)

    def get_stage_path(self, dataset, stage):
        """
        Get the database path for a specific stage of a dataset. Stages
        are raw, norm, sino, gridrec, etc...

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name

        Returns
        -------
        str
            database path
        """
        self.check_login()
        derivatives = self.get_derived_datasets(dataset)

        for i in range(len(derivatives)):
            if derivatives[i]['stage'] == stage:
                path = derivatives[i]['path']
                return path

        raise SPOTError('Stage %s in dataset %s does not exist' %
                        (stage, dataset))

    def get_file_location(self, dataset, stage):
        """
        Get the full location (system path) for a specific stage of a dataset.
        stages are raw, norm, sino, gridrec, etc...

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name

        Returns
        -------
        str
            file path
        """
        self.check_login()
        derivatives = self.get_derived_datasets(dataset)

        for i in range(len(derivatives)):
            if derivatives[i]['stage'] == stage:
                location = derivatives[i]['phyloc']
                return location

        raise SPOTError('Stage %s in dataset %s does not exist' %
                        (stage, dataset))

    def get_dataset_attributes(self, dataset, stage='raw', group='/'):
        """
        Get hdf5 attributes of a specified dataset. group can be a specified
        image within the hdf5 file, or default '/' for top level attributes

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        group : str
            group name of hdf5

        Returns
        -------
        json
            json reponse of attributes
        """

        path = self.get_stage_path(dataset, stage)
        params = {'group': group}
        r = self.post(self.SPOT_URL + '/hdf/attributes' + path, params=params)

        return self.check_response(r)

    def list_dataset_images(self, dataset, stage):
        """
        List images inside a given dataset

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name

        Returns
        -------
        json
            response of image list
        """

        path = self.get_stage_path(dataset, stage)
        r = self.post(self.SPOT_URL + '/hdf/listimages' + path)

        return self.check_response(r)

    def get_dataset_size(self, dataset, stage):
        """
        Get the file size of a specified dataset

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name

        Returns
        -------
        int
            dataset size in bytes
        """

        path = self.get_stage_path(dataset, stage)
        r = self.session.head(self.SPOT_URL + '/hdf/download' + path)
        head = r.headers
        if not 'content-length' in head: return 1
        size = float(head['content-length'])

        return size

    def get_raw_image(self, dataset, stage, image=None, index=None):
        """
        Download raw data from an image in a SPOT dataset

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        image : str
            (optional), name of image in dataset
        index : int
            (optional) index of image in dataset (one of index or image must be given)
        Returns
        -------
        ndarray
            2D ndarray of image data
        """

        images = list(self.list_dataset_images(dataset, stage))
        if image is None and index is None:
            raise ValueError('One of image or index must be given')
        elif image is None and index is not None:
            group = images[index]
        else:
            group = os.path.split(images[0])[0] + '/' + image

        r = self.stage_tape_2_disk(dataset, stage)
        path = self.get_stage_path(dataset, stage)
        params = {'group': group}
        r = self.post(self.SPOT_URL + '/hdf/rawdata' + path, params=params)
        r = self.check_response(r)
        return np.array(r['data'])


    def get_image_download_URLS(self, dataset, stage, image=None, index=None):
        """
        Get download URL's for a specific image in a SPOT dataset

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        image : str
            (optional), name of image in dataset
        index : int
            (optional) index of image in dataset (one of index or image must be given)
        Returns
        -------
        dict
            Dictionary with urls to images
        """

        images = list(self.list_dataset_images(dataset, stage))
        if image is None and index is None:
            raise ValueError('One of image or index must be given')
        elif image is None and index is not None:
            group = images[index]
        else:
            group = os.path.split(images[0])[0] + '/' + image

        r = self.stage_tape_2_disk(dataset, stage)
        path = self.get_stage_path(dataset, stage)
        params = {'group': group}
        r = self.post(self.SPOT_URL + '/hdf/image' + path, params=params)
        r = self.check_response(r)
        return r

    def get_image_as(self, dataset, stage, ext='tif', image=None, index=None):
        """
        Download an image in the specified format and return an array of the image

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        ext : str, optional
            extension for image type (tif or png)
        image : str
            (optional), name of image in dataset
        index : int
            (optional) index of image in dataset (one of index or image must be given)
        Returns
        -------
        ndarray
            2D ndarray of image data
        """

        r = self.get_image_download_URLS(dataset, stage, image=image, index=index)
        url = r['pnglocaion'] if ext == 'png' else r['tiflocaion']  # Careful when spot API fixes this spelling mistake
        r = self.get(url)
        img = Image.open(StringIO(r.content))
        return np.asarray(img)


    def download_image(self, dataset, stage, save_path=None, ext='tif', image=None, index=None):
        """
        Download and save a specific image in a dataset as png or tif image

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        save_path : str, optional
            Path to save the image
        ext : str, optional
            extension for image type (tif or png)
        image : str
            (optional), name of image in dataset
        index : int
            (optional) index of image in dataset (one of index or image must be given)
        """

        if ext not in ('png', 'tif'):
            raise ValueError('ext can only be png or tif')
        if image is None and index is None:
            raise ValueError('One of image or index must be given')

        if save_path is None:
            name = image.split('.')[0] if image is not None else dataset + '_{}'.format(index)
            save_path = os.path.join(os.path.expanduser('~'), '{}.{}'.format(name,ext))

        r = self.get_image_download_URLS(dataset, stage, image=image, index=index)
        url = r['pnglocaion'] if ext == 'png' else r['tiflocaion'] # Careful when spot API fixes this spelling mistake
        r = self.get(url)

        with open(save_path, 'w') as f:
            for chunk in r:
                f.write(chunk)

    def stage_tape_2_disk(self, dataset, stage):
        """
        Stage a dataset from tape to disk if needed

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        """

        path = self.get_stage_path(dataset, stage)
        r = self.check_response(self.post(self.SPOT_URL + '/hdf/stageifneeded' + path))
        # Wait for staging to finish
        while r['location'] == 'unknown' or r['location'] == 'staging':
            sleep(3)
            r = self.check_response(self.post(self.SPOT_URL + '/hdf/stageifneeded' + path))

        return r

    def download_dataset(self, dataset, stage, save_path=None):
        """
        Download a specified dataset

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        save_path : str
            Path and name to save file locally. If None name on SPOT is used and save in home directory
        """

        path = self.get_stage_path(dataset, stage)
        if save_path is None:
            save_path = os.path.join(os.path.expanduser('~'), path.split('/')[-1])

        r = self.stage_tape_2_disk(dataset, stage)
        r = self.get(self.SPOT_URL + '/hdf/download' + path, stream=True)

        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=64*1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        r.close()

        return self.check_login(r)

    def download_dataset_generator(self, dataset, stage, save_path=None, chunk_size=64*1024):
        """
        Download a dataset as a generator (yields the fraction downloaded)
        Useful to know the status of a download (for gui purposes)

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        save_path : str
            path and name to save file locally. If None name on SPOT is used and save in home directory
        chunk_size : int
            Chuck size of data in bytes

        Yields
        ------
        float
            Percent downloaded
        """

        path = self.get_stage_path(dataset, stage)
        if save_path is None:
            save_path = os.path.join(os.path.expanduser('~'), path.split('/')[-1])

        r = self.stage_tape_2_disk(dataset, stage)
        file_size = float(self.get_dataset_size(dataset, stage))
        r = self.get(self.SPOT_URL + '/hdf/download' + path, stream=True)

        with open(save_path, 'wb') as f:
            downloaded = 0.0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
                    downloaded += len(chunk)/file_size
                    yield downloaded
        r.close()

        return

    def transfer_2_nersc(self, dataset, stage, path, system):
        # TODO need to find a way to make a generator out of this, here or from cp/rsync
        """
        Transfer a dataset to NERSC

        Parameters
        ----------
        dataset : str
            name of dataset
        stage : str
            stage name
        path : str
            absolute destination path on NERSC
        """

        r = self.stage_tape_2_disk(dataset, stage)
        if r['location'] != 'staging' or r['ocation'] != 'unknown':
            r = self.rsync(r['location'], path, system)

        return r

class SPOTError(Exception):
    """Raised when SPOT gets angry"""
    pass


if __name__ == '__main__':
    import time
    from StringIO import StringIO
    from PIL import Image
    from matplotlib.pyplot import imshow, show, figure
    s = SpotClient()
    s.login('lbluque', '')
    # t = time.time()
    # img = s.get_raw_image('20160630_054009_prefire_3_0amp_scan7', 'raw',  index=0)
    # print 'Time: ', time.time() - t
    t = time.time()
    arr = s.get_image_as('20160630_054009_prefire_3_0amp_scan7', 'raw', ext='tif', index=0)
    print arr.shape
    imshow(arr)
    show()
    # for i in range(3):
    #     figure(i)
    #     imshow(arr[:, :, i])
    # show()
    print 'Time: ', time.time() - t
