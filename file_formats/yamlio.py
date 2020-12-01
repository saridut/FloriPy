#!/usr/bin/env python
'''
Module for reading from and writing to yaml files.

Useful reference for yaml file:
http://en.wikipedia.org/wiki/User:Baxter.brad/Drafts/YAML_Tutorial_and_Style_Guide
Useful methods for pyYAML (ruamel.yaml is built on pyYAML):
yaml.load(fh), yaml.load_all
yaml.dump(fh), yaml.dump_all
YAMLError(): exception
Loader and Dumper (defaults are fine, but may be required if  using LibYAML)
Use the RoundTrip versions of loader and dumper for preserving comments
and order of the dicts.
e.g. ruamel.yaml.load(fh_yaml, ruamel.yaml.RoundTripLoader)
    ruamel.yaml.dump(yaml_data, Dumper=ruamel.yaml.RoundTripDumper,
    explicit_start=True)
'''
import yaml

class ExplicitDumper(yaml.SafeDumper):
    """
    A dumper that will never emit aliases. Required make the output file human
    readable, at the expense of repeating redundant data.
    """
    def ignore_aliases(self, data):
        return True


def read(fn):
    '''Reads an yaml file.

    Paramters
    ---------
    fn : string
        Name of the yaml file to be read.

    Returns
    -------
    yamldict : dict
        Dict containing the contents of the yaml file.

    '''
    with open(fn, 'r') as fh_yaml:
        try:
            yamldict = yaml.load(fh_yaml)
        except yaml.YAMLError as yerr:
            print('Cannot load file {0}!\n'.format(fn), yerr)
            raise SystemExit('Goodbye...')
    return yamldict


def write(yamldict, fn, **kwargs):
    '''Writes to a yaml file.

    Parameters
    ----------
    yamldict : dict
        Dict to be written.
    fn : string
        Name of the file to which the dict `yamldict` will be written.
    kwargs: default_flow_style, indent
    '''
    with open(fn, 'w') as fh_yaml:
        yaml.dump(yamldict, stream=fh_yaml, explicit_start=True,
                Dumper=ExplicitDumper, **kwargs)


def append(yamldict, fn, default_flow_style=None, indent=4):
    '''Appends to an yaml file. If the yaml file does not exist, it is created.

    '''
    with open(fn, 'a') as fh_yaml:
        yaml.dump(yamldict, stream=fh_yaml, explicit_start=True,
                default_flow_style=default_flow_style, indent=indent,
                Dumper=ExplicitDumper)


def read_series(fn):
    '''This is a generator object returning each YAML document. It will exhaust
    when there are no more documents to read. This is useful for reading a
    series of yaml documents contained in a single file.

    '''
    fh_yaml = open(fn,'r')
    loader = yaml.Loader(fh_yaml)
    while loader.check_data():
        yield loader.get_data()
    fh_yaml.close()


if __name__ == '__main__':
    config = {'One':1, 'Two':2, 'Three':3, 'List':[1, 2, 2]}
    fn = 'test_yamlio.yaml'
    print('Dict to write')
    print('-------------')
    print(config)
    write(config, fn)
    print('Written to {0}'.format(fn))
    config_read = read(fn)
    print('Reading from file {0}'.format(fn))
    print(config)
    assert config == config_read

    #Testing append and read_series
    config = {'One':1, 'Two':2, 'Three':3, 'List':[1, 2, 2]}
    fn = 'test_yamlio.yaml'
    write(config, fn)
    config2 = {'One':'a', 'Two':'b', 'Three':'c', 'List':['a', 'b', 'c']}
    append(config2, fn)
    for each in read_series(fn):
        print(each)


