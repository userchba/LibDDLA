#!/usr/bin/env python3
"""Regenerate private declarations and public forwarding from documented headers.

Signature source: include/ddla/ddla.h. Scalar support and exceptional routing:
src/public_api_manifest.json. Edit only those policy fields in the manifest;
counts, template-parameter names and signature fields are regenerated.

Adding an API:
  1. Declare/document it in ddla.h (including public defaults/constraints).
  2. Add a manifest entry with name, header_overload (zero based among that
     name), supported_scalar_arguments, and policy. Preserve per-name overload
     indices when moving declarations; update them when reordering overloads.
  3. Run this script, then --check, CPU/CUDA build/routing and symbol checks.

The generator is development-time only. It adds no runtime dispatch layer and
uses the Python standard library (including the cluster's Python 3.6).
"""
from __future__ import print_function

import argparse
import collections
import json
from pathlib import Path
import re
import sys

HEADER = Path('include/ddla/ddla.h')
MANIFEST = Path('src/public_api_manifest.json')
BACKEND = Path('src/backend_api.h')
DISPATCH = Path('src/public_dispatch.cpp')
POLICIES = frozenset(('direct_backend_forward',
                      'cpu_only_rejection_without_backend_symbol',
                      'handle_stream_writer'))


def mask_comments_literals(text):
    """Keep offsets, but hide delimiters in C++ comments and string literals."""
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: re.sub(r'[^\n]', ' ', m.group()), text)


def compact(text):
    return re.sub(r'\s+', ' ', text).strip()


def matching(text, start, opening, closing):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('unbalanced {} at offset {}'.format(opening, start))


def split_top(text, separator=','):
    parts = []
    depth = {'<': 0, '(': 0, '[': 0, '{': 0}
    close = {'>': '<', ')': '(', ']': '[', '}': '{'}
    masked = mask_comments_literals(text)
    start = 0
    for i, ch in enumerate(masked):
        if ch in depth:
            depth[ch] += 1
        elif ch in close:
            depth[close[ch]] -= 1
        elif ch == separator and not any(depth.values()):
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def without_default(parameter):
    return split_top(parameter, '=')[0].strip()


def parameter_name(parameter):
    parameter = without_default(parameter)
    match = re.search(r'([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)?$', parameter)
    if not match:
        raise ValueError('parameter must be named: ' + parameter)
    return match.group(1)


def signature(decl, defaults=False, private=False, policy=None):
    parameters = decl['parameters'][:]
    returns = decl['returns']
    if private and policy:
        if policy['policy'] == 'handle_stream_writer':
            parameters = [p for p in parameters if 'DdlaHandle_t' not in p]
    if not defaults:
        parameters = [without_default(p) for p in parameters]
    return '{} {}({})'.format(returns, decl['name'], ', '.join(parameters))


def declarations(header):
    """Read the project's named, top-level template function declarations.

    This intentionally is not a general C++ parser. Definitions/traits are
    ignored; all exported handle-based computational declarations are checked
    for explicit policies, so unsupported syntax fails instead of disappearing.
    """
    masked = mask_comments_literals(header)
    found = []
    ordinals = collections.Counter()
    for match in re.finditer(r'\btemplate\s*<', masked):
        begin = masked.index('<', match.start())
        end = matching(masked, begin, '<', '>')
        # Traits and inline definitions have a top-level '{', declarations ';'.
        angle = paren = bracket = 0
        terminator = None
        for pos in range(end + 1, len(masked)):
            ch = masked[pos]
            if ch == '<': angle += 1
            elif ch == '>': angle -= 1
            elif ch == '(': paren += 1
            elif ch == ')': paren -= 1
            elif ch == '[': bracket += 1
            elif ch == ']': bracket -= 1
            elif not (angle or paren or bracket) and ch in ';{':
                terminator = pos
                break
        if terminator is None or masked[terminator] != ';':
            continue
        raw = compact(re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', ' ', header[end + 1:terminator]))
        if 'DdlaHandle_t' not in raw:
            continue
        masked_raw = mask_comments_literals(raw)
        angle = 0
        opening = None
        for pos, ch in enumerate(masked_raw):
            if ch == '<': angle += 1
            elif ch == '>': angle -= 1
            elif ch == '(' and angle == 0:
                opening = pos
                break
        if opening is None:
            raise ValueError('unsupported exported declaration: ' + raw)
        closing = matching(masked_raw, opening, '(', ')')
        if raw[closing + 1:].strip():
            raise ValueError('unsupported function suffix: ' + raw)
        prefix = raw[:opening].rstrip()
        name_match = re.search(r'([A-Za-z_]\w*)$', prefix)
        if not name_match:
            raise ValueError('unsupported function name: ' + raw)
        name = name_match.group(1)
        parameters = split_top(raw[opening + 1:closing])
        for parameter in parameters:
            parameter_name(parameter)
        templates = [compact(p) for p in split_top(header[begin + 1:end])]
        if any(not re.match(r'^(typename|class)\s+\w+(?:\s*=.*)?$', p) for p in templates):
            raise ValueError('only named scalar template parameters are supported: ' + name)
        names = [parameter_name(p) for p in templates]
        found.append(dict(name=name, returns=prefix[:name_match.start()].strip(),
                          parameters=parameters, templates=templates,
                          scalar_parameters=names, header_overload=ordinals[name]))
        ordinals[name] += 1
    return found


def read_contract(root):
    header = (root / HEADER).read_text()
    manifest = json.loads((root / MANIFEST).read_text())
    decls = declarations(header)
    manual = manifest['manual_public_templates']
    if len(manual) != len(set(manual)):
        raise ValueError('duplicate manual public template names')
    # Per-name overload indices keep unrelated declaration moves independent.
    policies = {}
    for entry in manifest['overloads']:
        name = entry['name']
        ordinal = entry['header_overload']
        key = (name, ordinal)
        if key in policies:
            raise ValueError('duplicate scalar policy: {}'.format(key))
        if entry['policy'] not in POLICIES:
            raise ValueError('unknown routing policy: ' + entry['policy'])
        policies[key] = entry
    generated = []
    for decl in decls:
        if decl['name'] in manual:
            continue
        key = (decl['name'], decl['header_overload'])
        if key not in policies:
            raise ValueError('missing scalar/routing policy for {}'.format(key))
        old = policies.pop(key)
        values = old['supported_scalar_arguments']
        if not values or len(values) != len(set(values)):
            raise ValueError('empty or duplicate scalar specializations: {}'.format(key))
        for value in values:
            if len(split_top(value)) != len(decl['scalar_parameters']):
                raise ValueError('wrong number of scalar arguments for {}: {}'.format(key, value))
        # Explicit instantiations spell void, while declarations preserve the
        # public SFINAE constraint. Fail on new return syntax needing review.
        returns = re.sub(r'\s+', '', decl['returns'])
        if returns not in ('void', 'std::size_t') and not (returns.startswith('typenamestd::enable_if<')
                                      and returns.endswith(',void>::type')):
            raise ValueError('unsupported public return type: ' + decl['returns'])
        entry = dict(index=len(generated), name=decl['name'],
                     header_overload=decl['header_overload'],
                     scalar_parameters=decl['scalar_parameters'],
                     public_signature=signature(decl),
                     private_signature=signature(decl, private=True,
                         policy=old),
                     supported_scalar_arguments=values, policy=old['policy'])
        generated.append((decl, entry))
    if policies:
        raise ValueError('scalar policies have no public declaration: {}'.format(sorted(policies)))
    manual_found = {d['name'] for d in decls if d['name'] in manual}
    if manual_found != set(manual):
        raise ValueError('manual public templates missing from header: {}'.format(set(manual) - manual_found))
    manifest['signature_source'] = str(HEADER)
    manifest['generated_fields'] = ['index', 'scalar_parameters', 'public_signature',
                                    'private_signature', 'public_backend_overloads',
                                    'public_concrete_specializations']
    manifest['manual_public_templates'] = manual
    manifest['public_backend_overloads'] = len(generated)
    manifest['public_concrete_specializations'] = sum(len(e['supported_scalar_arguments']) for d, e in generated)
    manifest['overloads'] = [e for d, e in generated]
    return generated, manifest


def template_line(decl, backend=False):
    parameters = [without_default(p) for p in decl['templates']]
    if backend:
        parameters.insert(0, 'DdlaBackend Backend')
    return 'template <{}>'.format(', '.join(parameters))


def backend_text(entries):
    lines = ['#ifndef DDLA_BACKEND_API_H', '#define DDLA_BACKEND_API_H', '',
             '// Generated by tools/generate_public_api.py; do not edit.',
             '// Signatures/docs: include/ddla/ddla.h. Scalar/routing policies: public_api_manifest.json.',
             '// This private header is never installed. Only implementation code selects Backend.',
             '#include <ddla/ddla.h>', '', 'namespace ddla {', '']
    for decl, entry in entries:
        lines += [template_line(decl, backend=True),
                  signature(decl, defaults=True, private=True, policy=entry) + ';', '']
    lines += ['} // namespace ddla', '', '#endif // DDLA_BACKEND_API_H', '']
    return '\n'.join(lines)


def dispatch_text(entries):
    lines = ['// Generated by tools/generate_public_api.py; do not edit.',
             '// Public signatures/docs: include/ddla/ddla.h. Scalar/routing policy: public_api_manifest.json.',
             '// Backend-specialized implementations and inner calls stay in their numerical translation units.',
             '#include "backend_api.h"', '#include "ddla_stream.h"', '#include <stdexcept>',
             '#include <string>', '', 'namespace ddla {', '']
    for decl, entry in entries:
        if entry['policy'] == 'handle_stream_writer':
            lines += ['// write_matrix has stream/lifetime handling in src/write_matrix.cpp.', '']
            continue
        handle = [parameter_name(p) for p in decl['parameters'] if 'DdlaHandle_t' in p]
        if len(handle) != 1:
            raise ValueError('forwarded routine needs exactly one handle: ' + decl['name'])
        handle = handle[0]
        arguments = ', '.join(parameter_name(p) for p in decl['parameters'])
        scalar = ', '.join(decl['scalar_parameters'])
        def call(backend):
            return '{}<DdlaBackend::{}, {}>({})'.format(decl['name'], backend, scalar, arguments)
        lines += [template_line(decl), signature(decl), '{', '#if DDLA_HAS_GPU',
                  '    if ({h} && {h}->backend == DdlaBackend::GPU)'.format(h=handle),
                  '        return ' + call('GPU') + ';']
        if entry['policy'] == 'cpu_only_rejection_without_backend_symbol':
            lines += ['    return ' + call('CPU') + ';', '#else', '    if (!' + handle + ')',
                      '        throw std::invalid_argument("' + decl['name'] + ': invalid DDLA handle");',
                      '    throw std::invalid_argument("' + decl['name'] + ': only the GPU backend is implemented");', '#endif']
        else:
            lines += ['#endif', '    return ' + call('CPU') + ';']
        lines += ['}', '']
        for concrete in entry['supported_scalar_arguments']:
            substitutions = dict(zip(decl['scalar_parameters'], split_top(concrete)))
            params = ', '.join(without_default(p) for p in decl['parameters'])
            for key, value in substitutions.items():
                params = re.sub(r'\b' + re.escape(key) + r'\b', lambda m: value, params)
            returns = 'std::size_t' if decl['returns'] == 'std::size_t' else 'void'
            lines.append('template {} {}<{}>({});'.format(returns, decl['name'], concrete, params))
        lines += ['']
    lines += ['} // namespace ddla', '']
    return '\n'.join(lines)


def generated_outputs(root):
    entries, manifest = read_contract(root)
    return {BACKEND: backend_text(entries), DISPATCH: dispatch_text(entries),
            MANIFEST: json.dumps(manifest, indent=2) + '\n'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='project root (default: generator parent directory)')
    parser.add_argument('--check', action='store_true', help='fail on stale generated files; never write')
    options = parser.parse_args(argv)
    try:
        outputs = generated_outputs(options.root)
        stale = [str(p) for p, content in outputs.items()
                 if not (options.root / p).exists() or (options.root / p).read_text() != content]
        if options.check:
            if stale:
                print('Stale public API outputs: ' + ', '.join(stale), file=sys.stderr)
                print('Run python3 tools/generate_public_api.py', file=sys.stderr)
                return 1
        else:
            for p, content in outputs.items():
                (options.root / p).write_text(content)
        manifest = json.loads(outputs[MANIFEST])
        print('Public API generation {}: {} overloads, {} scalar specializations'.format(
            'check passed' if options.check else 'updated', manifest['public_backend_overloads'],
            manifest['public_concrete_specializations']))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print('Public API generation failed: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
