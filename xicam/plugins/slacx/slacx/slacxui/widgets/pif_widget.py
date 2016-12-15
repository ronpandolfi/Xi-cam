from PySide import QtCore, QtGui
import pypif.obj as pifobj

from text_widgets import display_text, unit_indent

# TODO: Make an (editable?) PIF Widget

class PifWidget(QtGui.QTextEdit):
    
    def __init__(self,itm):
        t = self.display_pif(itm,unit_indent)
        super(PifWidget,self).__init__()
        self.setText(t) 

    def display_pif(self,itm,indent):
        t = indent + '(pypif.obj.System)'
        if itm.uid: 
            #string
            t += '<br>' + indent + 'uid: <br>{}'.format(display_text(itm.uid,indent+unit_indent))
        if isinstance(itm,pifobj.ChemicalSystem):
            if itm.chemical_formula: 
                #string
                t += '<br>' + indent + 'chemical_formula: <br>{}'.format(display_text(itm.chemical_formula,indent+unit_indent))
            if itm.composition is not None:
                #list of pypif.obj.Composition
                t += '<br>' + indent + 'composition: '
                t += '<br>' + indent + unit_indent + '(list)'
                for i,comp in zip(range(len(itm.composition)),itm.composition):
                    t += '<br>' + indent + unit_indent + '{}: <br>{}'.format(
                    i,self.display_comp(comp,indent+unit_indent+unit_indent))
        if itm.names is not None: 
            #list of string
            t += '<br>' + indent + 'names: '
            t += '<br>' + indent + unit_indent + '(list)'
            for i,nm in zip(range(len(itm.names)),itm.names):
                t += '<br>' + indent + unit_indent + '{}: <br>{}'.format(
                i, display_text(nm,indent+unit_indent+unit_indent))
        if itm.quantity: 
            #pypif.obj.Quantity
            t += '<br>' + indent + 'quantity: <br>{}'.format(self.display_qty(itm.quantity,indent+unit_indent))
        if itm.source: 
            #pypif.obj.Source
            t += '<br>' + indent + 'source: <br>{}'.format(self.display_src(itm.source,indent+unit_indent))
        if itm.preparation is not None: 
            #list of pypif.obj.ProcessStep
            t += '<br>' + indent + 'preparation: '
            t += '<br>' + indent + unit_indent + '(list)'
            for i,procstep in zip(range(len(itm.preparation)),itm.preparation):
                t += '<br>' + indent + unit_indent + '{}: <br>{}'.format(
                i, self.display_procstep(procstep,indent+unit_indent+unit_indent))
        if itm.properties is not None: 
            #list of pypif.obj.Property
            t += '<br>' + indent + 'properties: '
            t += '<br>' + indent + unit_indent + '(list)'
            for i,prop in zip(range(len(itm.properties)),itm.properties):
                t += '<br>' + indent + unit_indent + '{}: <br>{}'.format(
                i, self.display_prop(prop,indent+unit_indent+unit_indent))
        if itm.tags is not None: 
            #list of string
            t += '<br>' + indent + 'tags: '
            t += '<br>' + indent + unit_indent + '(list)'
            for i,tg in zip(range(len(itm.tags)),itm.tags):
                t += '<br>' + indent + unit_indent + '{}: <br>{}'.format(
                i, display_text(tg,indent+unit_indent+unit_indent))
        if itm.ids is not None: 
            #list of pypif.obj.Id
            t += '<br>' + indent + 'ids: '
            t += '<br>' + indent + unit_indent + '(list)'
            for i,id_ in zip(range(len(itm.ids)),itm.ids):
                t += '<br>' + indent + unit_indent + '{}: <br>{}'.format(
                i, self.display_id(id_,indent+unit_indent+unit_indent))
        if itm.sub_systems is not None: 
            #list of pypif.obj.System
            t += '<br>' + indent + 'sub_systems: '
            t += '<br>' + indent + unit_indent + '(list)'
            for i,sys in zip(range(len(itm.sub_systems)),itm.sub_systems):
                t += '<br>' + indent + unit_indent + '{}: <br>{}'.format(
                i, self.display_pif(sys,indent+unit_indent+unit_indent))
        return t    
    
    def display_comp(self,itm,indent):    
        t = indent + '(pypif.obj.Composition)'
        return t
    
    def display_qty(self,itm,indent):    
        t = indent + '(pypif.obj.Quantity)'
        return t

    def display_src(self,itm,indent):    
        t = indent + '(pypif.obj.Source)'
        return t

    def display_procstep(self,itm,indent):
        t = indent + '(pypif.obj.ProcessStep)'
        return t

    def display_prop(self,itm,indent):
        t = indent + '(pypif.obj.Property)'
        return t

    def display_id(self,id_,indent):
        t = indent + '(pypif.obj.Id)'
        return t


