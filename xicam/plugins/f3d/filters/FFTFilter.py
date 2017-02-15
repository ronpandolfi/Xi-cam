
class FFTFilter:

    def __init__(self):
        self.name = 'FFTFilter'
        self.index = -1

        # load clattr from RunnablePOCLFilter
        self.clattr = None
        self.atts = None #load attributes from RunnablePOCLFilter