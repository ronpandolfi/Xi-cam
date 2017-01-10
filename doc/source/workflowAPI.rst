.. highlight:: python
    :linenothreshold: 2


Workflow API Development Tutorial
=================================

What is the Workflow API?
-------------------------

The Xi-cam workflow API allows users to add their own functions into the Xi-cam processing environment. With only casual
experience in Python and no knowledge of Qt, you can give your custom analysis or processing technique an accessible
user interface. The benefits of this approach can include:

* Community exposure of your analysis technique
* Rapid GUI integration
* Explore results/output with Xi-cam's visualization tools
* Integration with remote data interfaces (SFTP, SPOT, Globus)
* Seamless execution of your function/workflow locally or remotely

How does it work?
-----------------

To develop a function for use in the workflow 'User/Developers' must provide 2 things:

* A dictionary defining a reference to the function and parameters which can be set in the dynamic interface
* The function, which must follow guidelines for its signature and output

Here is an example:

.. code-block:: python
    :linenos:

        def RickerWaveletCenterFind(minr, maxr, **workspace):
            '''
            This function takes two user-configurable parameters: minr and maxr.
            The workspace kwargs is passed along through each step of the workflow
            '''
            import numpy as np
            from pipeline import center_approx
            from scipy import signal
            from xicam import config

            # The function pulls data out of the workspace
            rawdata = workspace['dimg'].rawdata

            # ...and does some math...
            radii = np.arange(minr, maxr)
            maxval = 0
            center = np.array([0, 0], dtype=np.int)
            for i in range(len(radii)):
                w = center_approx.tophat2(radii[i], scale=1000)
                im2 = signal.fftconvolve(rawdata, w, 'same')
                if im2.max() > maxval:
                    maxval = im2.max()
                    center = np.array(np.unravel_index(im2.argmax(), rawdata.shape))

            # In this case, it modifies the global calibration parameters...
            config.activeExperiment.center = center

            # ...but also adds to the workflow workspace
            updates = {'center': center}

            # This function just merges the two dictionaries together; updates have preference
            workspace = updateworkspace(workspace, updates)

            # This function's changes ('updates') and the updated workspace are both returned
            return workspace, updates

A block of YAML markup is used to provide a reference to this function and how it is exposed:

.. code-block:: python
    :linenos:

        functionManifest = """
        Center Finding:
            - displayName:  Ricker Wavelets
              functionName: RickerWaveletCenterFind
              moduleName:   saxsfunctions
              functionType: PROCESS
              parameters:
                  - name:   Search minimum
                    type:   int
                    limits: [1,100000]
                    suffix: ' px'
                  - name:   Search maximum
                    type:   int
                    limits: [1,100000]
                    suffix: ' px'
        """

A workflow module must contain a ``functionManifest`` which describes its contents. The 'parameters' block is a list of
dictionary objects which can be read using PyQtGraph's Parameter.create() method
(see `PyQtGraph API <http://www.pyqtgraph.org/documentation/parametertree/parametertypes.html>`_).

Function Types
--------------

There are 4 categories of workflow functions. These function categories have slightly unique signatures:

INPUT
^^^^^
The function adds data into the workflow.
Example signature:

.. code-block:: python
    :linenos:

        def function([userArg1, userArg2, ...,] **workspace):
            ...
            return workspace, updates

OUTPUT
^^^^^^

The function writes data to file, remote resource, or database. These functions should operate on the ``updates``
dictionary argument. These functions return no values.
Example signature:

.. code-block:: python
    :linenos:

        def function([userArg1, userArg2, ...,] updates, **workspace):
            ...
            # No return value

PROCESS
^^^^^^^
The function performs processing/analysis and adds its result to the workspace.
Example signature:

.. code-block:: python
    :linenos:

        def function([userArg1, userArg2, ...,] **workspace):
            ...
            return workspace, updates

VISUALIZE
^^^^^^^^^
The function builds a Qt visualization from the data in ``workspace``. These functions return a QWidget class and
a ``tuple`` of the class arguments to be used with its construction.
Example signature:

.. code-block:: python
    :linenos:

        def function([userArg1, userArg2, ...,] updates, **workspace):
            ...
            return QWidgetClass, (widgetArg1, widgetArg2, ...)

