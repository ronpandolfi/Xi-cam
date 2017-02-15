
class MaskFilter:

    def __init__(self):
        self.name = 'MaskFilter'
        self.index = -1

        # load clattr from RunnablePOCLFilter
        self.clattr = None
        self.atts = None #load attributes from RunnablePOCLFilter