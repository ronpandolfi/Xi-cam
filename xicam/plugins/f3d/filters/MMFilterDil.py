
class MMFilterDil:

    def __init__(self):
        self.name = 'MMFilterDil'
        self.index = -1

        # load clattr from RunnablePOCLFilter
        self.clattr = None
        self.atts = None #load attributes from RunnablePOCLFilter