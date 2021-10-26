# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transmission/cluster_data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='transmission/cluster_data.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1ftransmission/cluster_data.proto\"a\n\x06secret\x12\x0f\n\x07\x64\x61taIds\x18\x01 \x03(\x05\x12\x10\n\x08\x64\x61taCIds\x18\x02 \x03(\t\x12\x0c\n\x04\x63Ids\x18\x03 \x03(\t\x12\x13\n\x0b\x63ipherTexts\x18\x04 \x03(\x0c\x12\x11\n\texponents\x18\x05 \x03(\x05\"C\n\nglobalDist\x12\x0f\n\x07\x64\x61taIds\x18\x01 \x03(\x05\x12\x12\n\ncipherText\x18\x02 \x03(\x0c\x12\x10\n\x08\x65xponent\x18\x03 \x03(\x05\x32\x38\n\x10safeTransmission\x12$\n\x0c\x65xchangeData\x12\x07.secret\x1a\x0b.globalDistb\x06proto3'
)




_SECRET = _descriptor.Descriptor(
  name='secret',
  full_name='secret',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dataIds', full_name='secret.dataIds', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dataCIds', full_name='secret.dataCIds', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cIds', full_name='secret.cIds', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cipherTexts', full_name='secret.cipherTexts', index=3,
      number=4, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='exponents', full_name='secret.exponents', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=35,
  serialized_end=132,
)


_GLOBALDIST = _descriptor.Descriptor(
  name='globalDist',
  full_name='globalDist',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dataIds', full_name='globalDist.dataIds', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cipherText', full_name='globalDist.cipherText', index=1,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='exponent', full_name='globalDist.exponent', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=134,
  serialized_end=201,
)

DESCRIPTOR.message_types_by_name['secret'] = _SECRET
DESCRIPTOR.message_types_by_name['globalDist'] = _GLOBALDIST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

secret = _reflection.GeneratedProtocolMessageType('secret', (_message.Message,), {
  'DESCRIPTOR' : _SECRET,
  '__module__' : 'transmission.cluster_data_pb2'
  # @@protoc_insertion_point(class_scope:secret)
  })
_sym_db.RegisterMessage(secret)

globalDist = _reflection.GeneratedProtocolMessageType('globalDist', (_message.Message,), {
  'DESCRIPTOR' : _GLOBALDIST,
  '__module__' : 'transmission.cluster_data_pb2'
  # @@protoc_insertion_point(class_scope:globalDist)
  })
_sym_db.RegisterMessage(globalDist)



_SAFETRANSMISSION = _descriptor.ServiceDescriptor(
  name='safeTransmission',
  full_name='safeTransmission',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=203,
  serialized_end=259,
  methods=[
  _descriptor.MethodDescriptor(
    name='exchangeData',
    full_name='safeTransmission.exchangeData',
    index=0,
    containing_service=None,
    input_type=_SECRET,
    output_type=_GLOBALDIST,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SAFETRANSMISSION)

DESCRIPTOR.services_by_name['safeTransmission'] = _SAFETRANSMISSION

# @@protoc_insertion_point(module_scope)