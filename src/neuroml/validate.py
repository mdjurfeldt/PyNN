import sys
from lxml import etree
with open("/home/andrew/dev/libNeuroML/neuroml/nml/NeuroML_v2beta.xsd") as fp:
    xmlschema_doc = etree.parse(fp)
xmlschema = etree.XMLSchema(xmlschema_doc)
with open(sys.argv[1]) as fp:
    doc = etree.parse(fp)
xmlschema.assertValid(doc)