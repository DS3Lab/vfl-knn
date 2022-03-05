# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: transmission/tenseal_mi/tenseal_mi_aggr_server.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='transmission/tenseal_mi/tenseal_mi_aggr_server.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n4transmission/tenseal_mi/tenseal_mi_aggr_server.proto\"J\n\x0bmi_aggr_msg\x12\x13\n\x0b\x63lient_rank\x18\x01 \x01(\x05\x12\t\n\x01k\x18\x02 \x01(\x05\x12\x0e\n\x06groups\x18\x03 \x03(\x05\x12\x0b\n\x03msg\x18\x04 \x01(\x0c\"5\n\rmi_aggr_top_k\x12\x13\n\x0b\x63lient_rank\x18\x01 \x01(\x05\x12\x0f\n\x07ranking\x18\x02 \x03(\x05\x32>\n\x13MIAggrServerService\x12\'\n\x07\x61ggr_mi\x12\x0c.mi_aggr_msg\x1a\x0e.mi_aggr_top_kb\x06proto3'
)




_MI_AGGR_MSG = _descriptor.Descriptor(
  name='mi_aggr_msg',
  full_name='mi_aggr_msg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_rank', full_name='mi_aggr_msg.client_rank', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='k', full_name='mi_aggr_msg.k', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='groups', full_name='mi_aggr_msg.groups', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='msg', full_name='mi_aggr_msg.msg', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
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
  serialized_start=56,
  serialized_end=130,
)


_MI_AGGR_TOP_K = _descriptor.Descriptor(
  name='mi_aggr_top_k',
  full_name='mi_aggr_top_k',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_rank', full_name='mi_aggr_top_k.client_rank', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ranking', full_name='mi_aggr_top_k.ranking', index=1,
      number=2, type=5, cpp_type=1, label=3,
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
  serialized_start=132,
  serialized_end=185,
)

DESCRIPTOR.message_types_by_name['mi_aggr_msg'] = _MI_AGGR_MSG
DESCRIPTOR.message_types_by_name['mi_aggr_top_k'] = _MI_AGGR_TOP_K
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

mi_aggr_msg = _reflection.GeneratedProtocolMessageType('mi_aggr_msg', (_message.Message,), {
  'DESCRIPTOR' : _MI_AGGR_MSG,
  '__module__' : 'transmission.tenseal_mi.tenseal_mi_aggr_server_pb2'
  # @@protoc_insertion_point(class_scope:mi_aggr_msg)
  })
_sym_db.RegisterMessage(mi_aggr_msg)

mi_aggr_top_k = _reflection.GeneratedProtocolMessageType('mi_aggr_top_k', (_message.Message,), {
  'DESCRIPTOR' : _MI_AGGR_TOP_K,
  '__module__' : 'transmission.tenseal_mi.tenseal_mi_aggr_server_pb2'
  # @@protoc_insertion_point(class_scope:mi_aggr_top_k)
  })
_sym_db.RegisterMessage(mi_aggr_top_k)



_MIAGGRSERVERSERVICE = _descriptor.ServiceDescriptor(
  name='MIAggrServerService',
  full_name='MIAggrServerService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=187,
  serialized_end=249,
  methods=[
  _descriptor.MethodDescriptor(
    name='aggr_mi',
    full_name='MIAggrServerService.aggr_mi',
    index=0,
    containing_service=None,
    input_type=_MI_AGGR_MSG,
    output_type=_MI_AGGR_TOP_K,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_MIAGGRSERVERSERVICE)

DESCRIPTOR.services_by_name['MIAggrServerService'] = _MIAGGRSERVERSERVICE

# @@protoc_insertion_point(module_scope)