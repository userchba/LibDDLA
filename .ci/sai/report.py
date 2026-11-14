#!/usr/bin/env python3
"""Keep missing tests, skips, process failures and Slurm failures visible."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def stage_result(folder, job):
    result = {'job': job, 'passed': 0, 'failed': 0, 'skipped': 0, 'total': 0,
              'cases': [], 'problems': [], 'success': False}
    try:
        inventory = json.loads((folder / 'ctest-inventory.json').read_text())['tests']
        expected = [case['name'] for case in inventory]
        if not expected or len(expected) != len(set(expected)):
            raise ValueError('Empty or duplicate CTest inventory')
        root = ET.parse(folder / 'ctest.xml').getroot()
        cases = list(root.iter('testcase'))
        names = [case.get('name') for case in cases]
        if len(names) != len(set(names)) or set(names) != set(expected):
            result['problems'].append('JUnit does not cover the exact CTest inventory')
        for case in cases:
            state = 'passed'
            if case.find('failure') is not None or case.find('error') is not None:
                state = 'failed'
            elif case.find('skipped') is not None or case.get('status') in ('notrun', 'disabled'):
                state = 'skipped'
            result[state] += 1
            result['cases'].append({'name': case.get('name'), 'status': state,
                                    'seconds': float(case.get('time', 0))})
        result['total'] = len(cases)
        for key, actual in [('tests', result['total']), ('failures', result['failed']),
                            ('skipped', result['skipped'])]:
            if key in root.attrib and int(root.get(key)) != actual:
                result['problems'].append(f'Inconsistent JUnit {key} count')
        if int(root.get('disabled', 0)):
            result['problems'].append('JUnit contains disabled tests')
        for name in ('exitcode.txt', 'ctest-exitcode.txt'):
            code = int((folder / name).read_text().strip())
            result[name] = code
            if code:
                result['problems'].append(f'{name}: {code}')
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        result['problems'].append(f'Incomplete evidence: {error}')
    if job.get('state') != 'COMPLETED' or job.get('exit_code') != '0:0':
        result['problems'].append('Slurm did not finish COMPLETED with exit 0:0')
    result['success'] = (result['total'] > 0 and not result['problems']
                         and result['failed'] == result['skipped'] == 0)
    return result


def generate(results, state):
    stages = {kind: stage_result(results / kind, state.get('jobs', {}).get(kind, {}))
              for kind in ('gpu4', 'gpu6')}
    summary = {'request': state.get('request', {}), 'stages': stages,
               'success': all(stage['success'] for stage in stages.values())}
    (results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# SAI CUDA 12.9 validation', '',
             f"Source: `{state.get('request', {}).get('source_sha', 'unknown')}`", '',
             '| Stage | Passed | Failed | Skipped | Total | Slurm |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for kind, stage in stages.items():
        lines.append(f"| {kind} | {stage['passed']} | {stage['failed']} | {stage['skipped']} | "
                     f"{stage['total']} | {stage['job'].get('state', 'missing')} |")
    for kind, stage in stages.items():
        lines += ['', f'## {kind}', '']
        lines += ['- ' + problem for problem in stage['problems']]
        lines += [f"- {case['status']}: `{case['name']}`" for case in stage['cases'] if case['status'] != 'passed']
    lines += ['', 'All failures are retained. Known failures are not converted to passes.', '']
    (results / 'summary.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('state', type=Path)
    args = parser.parse_args()
    summary = generate(args.results, json.loads(args.state.read_text()))
    raise SystemExit(0 if summary['success'] else 1)
