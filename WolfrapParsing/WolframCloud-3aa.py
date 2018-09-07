from urllib import urlencode
from urllib2 import urlopen


class WolframCloud:
    @staticmethod
    def wolfram_cloud_call(self, **args):
        arguments = dict([(key, arg) for key, arg in args.iteritems()])
        result = urlopen("http://www.open.wolframcloud.com/objects/6f3ed7a3-c86e-4ef6-b1a1-5998f4ed15b1", urlencode(arguments))
        return result.read()

    def call(self, x):
        textresult =  self.wolfram_cloud_call(x=x)
        return textresult


if __name__ == "__main__":
    wolf = WolframCloud({})
    print(wolf.wolfram_cloud_call())
