

from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, \
    Literal, Number, Operator, Other, Punctuation, Text, Generic, \
    Whitespace


class GraphiteStyle(Style):
    background_color = "#2B2B2B"
    default_style = ""

    styles = {
        Keyword: 'bold #CA7732',
        Keyword.Constant: 'bold #CA7732',
        Keyword.Namespace: 'bold #CA7732',
        Keyword.Pseudo: 'bold #CA7732',
        Name: '#A8B6c4',
        Name.Builtin: 'bold #A8B6c4',
        Name.Class: 'bold #A8B6c4',
        Name.Function: 'bold #A8B6c4',
        Name.Decorator: '#A8B6c4',
        Literal: '#9BB855',
        Number: '#6796BA',
        String: '#9BB855',
        String.Escape: 'bold #C07432',
        Operator:               '#eeeeee',
        Operator.Word:          'bold #eeeeee',
        Punctuation:            '#eeeeee',
        Comment: '#7F7F7D',
    }


a = 2
'\n'
"""
"""
# *
