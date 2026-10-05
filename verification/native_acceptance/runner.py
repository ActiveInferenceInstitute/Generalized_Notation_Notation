"""Fresh four-case native acceptance per backend; no work runs at import.

Run only after peer review and root resource admission. All evidence is retained
in a unique persistent scratch directory. One aggregate monotonic deadline spans
admission, probes, every pipeline, artifact validation and cleanup. No retries,
shortened models, source repair or implicit dependency installation occur here.
"""
from __future__ import annotations
import argparse
import hashlib
import types
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

CODE = Path(__file__).resolve().parent
BASE = Path(os.environ["GNN_NATIVE_VERIFY_WORKSPACE"]).absolute()
ROOT = Path(os.environ["GNN_NATIVE_VERIFY_SOURCE_ROOT"]).absolute()
SOURCE = '88f24cb7cb3026ab6da66eed0d754037f763865e'
ARTIFACT = 'a39e8720a0fb2f9a5e2deeab4ddc19394fe99e4a'
WHEEL = '36e71bebe855711b12751d5d58feaeac5d45f53c12e21eb5481e80405c41b907'
COMMON = CODE / 'common.py'
COMMON_SHA = '74d061a42bad33a5c53d3be42f797a888acb19003bb2e5e75cb1f0bcbd85935e'
WORKER = CODE / 'artifacts.py'
CASES = {
    'numpyro': [('discrete/multi_armed_bandit.md', 30), ('discrete/actinf_pomdp_agent.md', 30),
                ('multiagent/multi_agent_coordination.md', 20), ('precision/precision_weighted.md', 30)],
    'rxinfer': [('multiagent/multi_agent_coordination.md', 20), ('multiagent/stigmergic_swarm.md', 30),
                ('pomdp_gridworld/pomdp_gridworld_3x3.md', 15), ('discrete/two_state_bistable.md', 20)],
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_common():
    require(not COMMON.is_symlink(), 'Private common helper symlink refused')
    raw = COMMON.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == COMMON_SHA, 'Private reviewed common helper changed')
    module = types.ModuleType('reported_native_common')
    module.__file__ = str(COMMON)
    exec(compile(raw, str(COMMON), 'exec', dont_inherit=True), module.__dict__)
    return module


def read_json(path):
    require(path.is_file() and not path.is_symlink(), 'Missing/symlink receipt:' + str(path))
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON:' + value)))


def retained_envelope(command, cwd, directory, name, deadline, token, timeout):
    from gnn.execute.subprocess_envelope import run_subprocess_envelope
    require(time.monotonic() < deadline and not token.cancelled, 'Budget/resource exhausted before dispatch')
    envelope = run_subprocess_envelope(command, cwd=cwd, env={'GNN_SANDBOX': 'off', 'GKSwstype': '100', 'JULIA_PKG_OFFLINE': 'true', 'JULIA_PKG_PRECOMPILE_AUTO': '0', 'TMPDIR': str(directory / 'tmp')},
        timeout=min(timeout, deadline - time.monotonic()), deadline_monotonic=deadline,
        cancel_token=token, sandbox=False)
    (directory / (name + '.stdout.log')).write_text(envelope.pop('stdout'))
    (directory / (name + '.stderr.log')).write_text(envelope.pop('stderr'))
    (directory / (name + '.envelope.json')).write_text(json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return envelope


def git(command, deadline, token):
    from gnn.execute.subprocess_envelope import run_subprocess_envelope
    require(time.monotonic() < deadline and not token.cancelled, 'Budget/resource exhausted before source inspection')
    receipt = run_subprocess_envelope(['git', '-c', 'core.fsmonitor=false', *command], cwd=ROOT, timeout=min(15, deadline - time.monotonic()), deadline_monotonic=deadline, cancel_token=token, sandbox=False)
    require(receipt.get('success') is True and receipt.get('cleanup_verified') is True and receipt.get('streams_drained') is True, 'Bounded Git source inspection failed')
    return receipt['stdout']


def bookend(deadline, token):
    head = git(['rev-parse', 'HEAD'], deadline, token).strip()
    require(git(['status', '--porcelain', '--untracked-files=no'], deadline, token) == '', 'ROOT tracked files are dirty')
    names = git(['ls-files', '-z'], deadline, token).split('\x00')
    inventory = {}
    for index, name in enumerate(filter(None, names)):
        if index % 128 == 0:
            require(time.monotonic() < deadline and not token.cancelled, 'Budget/resource exhausted during source bookend')
        path = ROOT / name
        require(path.is_relative_to(ROOT), 'Tracked path escapes ROOT')
        if path.is_symlink():
            inventory[name] = hashlib.sha256(('symlink:' + os.readlink(path)).encode()).hexdigest()
        else:
            require(path.is_file(), 'Missing tracked file:' + name)
            inventory[name] = digest(path)
    require(len(inventory) >= 3154, 'Verification branch omitted released tracked files')
    binding = read_json(CODE / 'source-binding.json')
    require(binding['expected_release_head'] == '09c0f346a10b835f4441ff61afa120899c4e9b01', 'Wrong reviewed release source')
    git(['merge-base', '--is-ancestor', binding['expected_release_head'], head], deadline, token)
    for name, expected_sha in binding['release_owner_inventory'].items():
        require(inventory.get(name) == expected_sha, 'Reviewed release owner changed:' + name)
    require(len(binding['release_owner_inventory']) == 745, 'Expected complete 745-owner source binding')
    return {'head': head, 'files': inventory, 'inventory_sha256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()}


def artifact_inventory(output):
    values = {}
    for path in sorted(output.rglob('*')):
        require(not path.is_symlink(), 'Output artifact symlink refused')
        if path.is_dir():
            continue
        require(path.is_file() and path.resolve().is_relative_to(output.resolve()), 'Nonregular/escaped output artifact')
        values[str(path.relative_to(output))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return values


def expected_model_id(relative):
    require(isinstance(relative, str) and relative == Path(relative).as_posix()
            and not Path(relative).is_absolute() and '..' not in Path(relative).parts,
            'Expected relative source path is invalid')
    stem = re.sub(r'[^A-Za-z0-9_-]', '_', Path(relative).stem)
    return stem + '-' + hashlib.sha256(relative.encode()).hexdigest()[:12]


def verify_selection(context, detail, source, input_root, relative, source_hash, backend):
    """Independently bind path-based identity; matching two forged IDs is insufficient."""
    model_id = expected_model_id(relative)
    require(not source.is_symlink() and source.is_file()
            and source.absolute() == input_root.absolute() / relative
            and digest(source) == source_hash, 'Selected source path/bytes differ')
    require(Path(context['input_root']).absolute() == input_root.absolute()
            and context['frameworks'] == [backend], 'Run input root/backend differs')
    require(len(context['models']) == 1, 'Selection does not contain exactly one authored model')
    model = context['models'][0]
    for selected, path_key, relative_key, sha_key in ((model, 'source_path', 'relative_path', 'sha256'),
                                                     (detail, 'source_path', 'source_relative_path', 'source_sha256')):
        require(Path(selected[path_key]).absolute() == source.absolute()
                and selected[relative_key] == relative and selected[sha_key] == source_hash
                and selected['model_id'] == model_id, 'Selected source path/model ID/SHA differs')
    require(model['artifact_stem'] == Path(relative).stem and detail['framework'] == backend,
            'Selected artifact stem/backend differs')
    return model_id


def require_bound_file(output, path, verified):
    """Require actual regular evidence bytes to belong to the finalized step map."""
    require(path.is_relative_to(output), 'Candidate artifact path escapes output')
    require(not any(part.is_symlink() for part in (path, *path.parents)), 'Artifact ancestor symlink refused')
    relative = path.relative_to(output).as_posix()
    require(relative in verified and path.is_file() and digest(path) == verified[relative],
            'Artifact is absent from finalized hashes or bytes differ: ' + relative)
    return relative


def finalized_pipeline(output, envelope, canonical, source_hash, configured,
                       source, input_root, relative, backend):
    summary_path = output / '00_pipeline_summary/pipeline_execution_summary.json'
    summary = read_json(summary_path)
    common = load_common()
    status = common.validate_pipeline_process(envelope, summary, canonical)
    session_path = output / '00_pipeline_summary/run_session.json'
    require(digest(session_path) == summary['run_session_sha256'], 'Session bytes do not match finalized summary')
    session = read_json(session_path)
    require(session['session_id'] == summary['run_id'] and session['final_status'] == status and session['evidence_integrity'] == summary['evidence_integrity'], 'Session final identity/status/evidence differs')
    require([unit['unit_id'] for unit in session['units']] == summary['planned_steps'], 'Session step plan differs')
    steps = {step['script_name']: step for step in summary['steps']}
    verified = {}
    for unit in session['units']:
        require(unit['status'] == 'DONE' and unit['input_identity']['run_id'] == summary['run_id'], 'Unfinished or unrelated session unit')
        expected = {item['path']: item['sha256'] for item in steps[unit['unit_id']]['artifacts']}
        require(unit['artifact_hashes'] == expected, 'Session artifact hashes differ')
        for artifact_relative, expected_sha in expected.items():
            path = output / artifact_relative
            require(not Path(artifact_relative).is_absolute() and '..' not in Path(artifact_relative).parts and path.resolve().is_relative_to(output.resolve()), 'Final artifact path escapes output')
            require(not any(part.is_symlink() for part in (path, *path.parents)) and path.is_file() and digest(path) == expected_sha, 'Final artifact bytes differ')
            verified[artifact_relative] = expected_sha
    context = read_json(output / '00_pipeline_summary/run_context.json')
    require(context['run_id'] == summary['run_id'] and Path(context['output_root']).absolute() == output.absolute(), 'Current-run context/output root differs')
    require(json.loads(context['config_json']) == configured, 'Caller resolved configuration differs')
    require(len(context['models']) == 1 and context['models'][0]['sha256'] == source_hash, 'Selected model/source binding differs')
    execution = read_json(output / '12_execute_output/summaries/execution_summary.json')
    require(execution['run_id'] == summary['run_id'] and execution['success'] is True and execution['status'] == 'success', 'Execution summary not current successful')
    require(execution['successful_executions'] == 1 and execution['failed_executions'] == execution['skipped_executions'] == 0, 'Native required execution failed/skipped')
    require(execution['attempted_scripts'] == execution['total_scripts'] == 1, 'Unexpected execution attempt count')
    details = execution['execution_details']
    require(len(details) == 1 and details[0]['success'] is True and not details[0].get('skipped'), 'Missing/duplicate/unaccepted execution detail')
    detail = details[0]
    verify_selection(context, detail, source, input_root, relative, source_hash, backend)
    require(detail['cleanup_verified'] is True and detail['streams_drained'] is True and detail.get('cancelled') is False, 'Native execution cleanup/cancellation unverified')
    return {'run_id': summary['run_id'], 'model': context['models'][0], 'status': status, 'required_step_statuses': {str(canonical[name]): step['status'] for name, step in steps.items()}, 'verified_pipeline_artifact_files': len(verified), 'verified_pipeline_artifacts': verified, 'session_sha256': digest(session_path), 'summary_sha256': digest(summary_path), 'execution_summary_sha256': digest(output / '12_execute_output/summaries/execution_summary.json'), 'execution_detail': detail, 'resolved_config_sha256': hashlib.sha256(context['config_json'].encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=tuple(CASES), required=True)
    parser.add_argument('--split', choices=('311', '312'), default='311')
    parser.add_argument('--expected-root-head', required=True)
    parser.add_argument('--aggregate-budget', type=float, default=3600)
    parser.add_argument('--case-budget', type=float, default=900)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--admission-only', action='store_true')
    args = parser.parse_args()
    require(math.isfinite(args.aggregate_budget) and args.aggregate_budget > 0 and math.isfinite(args.case_budget) and args.case_budget > 0, 'Budgets must be finite and positive')
    started = time.monotonic()
    deadline = started + args.aggregate_budget
    root = BASE / 'evidence' / ('reported-native-' + args.backend + '-' + args.split + '-' + uuid.uuid4().hex[:8])
    root.mkdir(exist_ok=False)
    (root / 'tmp').mkdir()
    record = {'accepted': False, 'backend': args.backend, 'split': args.split, 'source_commit': SOURCE, 'artifact_commit': ARTIFACT, 'wheel_sha256': WHEEL, 'expected_root_head': args.expected_root_head, 'runner_sha256': digest(Path(__file__)), 'common_sha256': COMMON_SHA, 'worker_sha256': args.worker_sha256, 'receipt_directory': str(root), 'aggregate_budget_seconds': args.aggregate_budget, 'case_budget_seconds': args.case_budget, 'cases': [], 'admission_only': args.admission_only}
    token = None
    before = None
    try:
        require(sys.flags.isolated == 1, 'Run with ordinary installed python -I')
        for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
            os.environ.pop(name, None)
        require(shutil.disk_usage(BASE).free >= 6 * 1024**3, 'RESOURCE_HOLD: free storage below six GiB launch admission')
        common = load_common()
        token = common.StorageToken()
        require(not token.cancelled, 'Authoritative four GiB storage floor exhausted')
        require(not WORKER.is_symlink() and digest(WORKER) == args.worker_sha256, 'Artifact validator changed before launch')
        before = bookend(deadline, token)
        require(before['head'] == args.expected_root_head, 'ROOT HEAD differs from admitted revision')
        (root / 'root-before.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
        binding, admission = common.verify_wheel(BASE, args.split, WHEEL, ARTIFACT)
        record['ordinary_wheel_admission'] = admission
        for name, expected_sha in binding['python_inventory'].items():
            require(digest(ROOT / 'src' / name) == expected_sha, 'ROOT source does not match admitted ordinary wheel:' + name)
        record['production_python_files_bound'] = len(binding['python_inventory'])
        from gnn.pipeline.step_registry import STEPS
        canonical = {step.script_name: int(step.script_stem.split('_', 1)[0]) for step in STEPS}
        require(len(canonical) == 25 and set(canonical.values()) == set(range(25)), 'Unexpected canonical step registry')
        from dataclasses import asdict
        from gnn.utils.runtime_safety.framework_availability import check_framework
        readiness = check_framework(args.backend, timeout=min(60, deadline - time.monotonic()), deadline_monotonic=deadline, cancel_token=token)
        record['readiness'] = asdict(readiness)
        require(readiness.available and readiness.cleanup_verified is True and readiness.streams_drained is True, 'Bounded native readiness is unavailable/unverified')
        require(not token.cancelled, 'Resource exhausted after readiness')
        if not args.admission_only:
            for relative, timesteps in CASES[args.backend]:
                case = root / Path(relative).stem
                case.mkdir()
                (case / 'tmp').mkdir()
                item = {'accepted': False, 'source_relative_path': 'input/gnn_files/' + relative, 'declared_timesteps': timesteps, 'overrides': {}, 'backend': args.backend}
                record['cases'].append(item)
                case_started = time.monotonic()
                case_deadline = min(deadline, case_started + args.case_budget)
                try:
                    require(not token.cancelled and case_started < deadline, 'Aggregate budget/resource exhausted before selected case')
                    source_dir = case / 'input/gnn_files'
                    source = source_dir / relative
                    source.parent.mkdir(parents=True)
                    committed = git(['show', SOURCE + ':input/gnn_files/' + relative], case_deadline, token).encode()
                    require((ROOT / 'input/gnn_files' / relative).read_bytes() == committed, 'Source changed since admitted content commit')
                    matches = re.findall(r'^num_timesteps\s*:\s*(\d+)', committed.decode(), flags=re.MULTILINE)
                    require(matches == [str(timesteps)], 'Authored timesteps differ from reviewed case')
                    source.write_bytes(committed)
                    source_hash = digest(source)
                    item['source_sha256'] = source_hash
                    output = case / 'output'
                    configured = {'pipeline': {'timeout': {'step': min(args.case_budget, case_deadline - time.monotonic()), 'total': min(args.case_budget, case_deadline - time.monotonic())}, 'parallel': {'enabled': False}}, 'testing_matrix': {'enabled': False}}
                    # Equal explicit budgets are fixed once before launch, then
                    # shared across configured pipeline and subsequent validator.
                    budget = min(int(args.case_budget), int(case_deadline - time.monotonic()) - 60)
                    require(budget > 0, 'No remaining pipeline budget after explicit60second artifact-verification reserve')
                    item['artifact_verification_reserve_seconds'] = 60
                    configured['pipeline']['timeout'] = {'step': budget, 'total': budget}
                    config = case / 'input/config.yaml'
                    config.write_text(json.dumps(configured, indent=2, sort_keys=True) + '\n')
                    item.update(config_sha256=digest(config), caller_configuration=configured, requested_inference_iterations=20 if args.backend == 'rxinfer' else None)
                    command = [sys.executable, '-I', '-m', 'gnn.main', '--target-dir', str(source_dir), '--output-dir', str(output), '--only-steps', '3,11,12,16,20', '--skip-steps', '0,1,2,13', '--skip-llm', '--frameworks', args.backend, '--serialize-preset', 'minimal', '--timeout', str(max(1, int(budget)))]
                    item['command'] = command
                    envelope = retained_envelope(command, case, case, 'pipeline', case_deadline, token, budget)
                    item['pipeline_envelope'] = envelope
                    require(not token.cancelled, 'Resource exhausted during pipeline')
                    item['finalization'] = finalized_pipeline(output, envelope, canonical, source_hash, configured, source, source_dir, relative, args.backend)
                    require(item['finalization']['execution_detail']['framework'] == args.backend, 'Execution backend differs from selected case')
                    require(digest(config) == item['config_sha256'], 'Pipeline modified caller configuration')
                    scripts = sorted((output / '11_render_output').rglob('*_' + args.backend + ('.jl' if args.backend == 'rxinfer' else '.py')))
                    require(len(scripts) == 1 and not scripts[0].is_symlink(), 'Missing/duplicate rendered native script')
                    results = sorted((output / '12_execute_output').rglob('*simulation_results.json'))
                    native = [path for path in results if '/simulation_data/' in str(path) and read_json(path).get('schema_version') in ({'rxinfer_simulation_v1', 'rxinfer_stigmergic_swarm_v1'} if args.backend == 'rxinfer' else {'numpyro_simulation_v1'})]
                    require(len(native) == 1, 'Missing/duplicate native result JSON')
                    verified = item['finalization']['verified_pipeline_artifacts']
                    item['bound_script_relative_path'] = require_bound_file(output, scripts[0], verified)
                    item['bound_native_relative_path'] = require_bound_file(output, native[0], verified)
                    analysis_candidates = sorted((output / '16_analysis_output').rglob('*'))
                    selected_analysis = [path for path in analysis_candidates if path.suffix.lower() in {'.png', '.gif'}]
                    require(bool(selected_analysis), 'No native analysis plot artifacts')
                    item['bound_analysis_files'] = {require_bound_file(output, path, verified): verified[path.relative_to(output).as_posix()] for path in selected_analysis}
                    item.update(script_sha256=digest(scripts[0]), native_result_sha256=digest(native[0]), native_result_path=str(native[0]))
                    identity = item['finalization']['execution_detail'].get('script_identity')
                    require(isinstance(identity, dict) and identity.get('sha256') == item['script_sha256'] and Path(identity['path']).resolve() == scripts[0].resolve(), 'Executed/rendered script identity differs')
                    require(item['finalization']['execution_detail'].get('attempts_started') == 1 and item['finalization']['execution_detail'].get('return_code') == 0, 'Native execution failed or retried')
                    require(digest(WORKER) == args.worker_sha256, 'Artifact validator changed during pipeline')
                    supplemental = case / 'supplemental-analysis'
                    validation_command = [sys.executable, '-I', str(WORKER), '--backend', args.backend, '--native', str(native[0]), '--script', str(scripts[0]), '--analysis', str(output / '16_analysis_output'), '--supplemental', str(supplemental), '--timesteps', str(timesteps), '--source', str(source), '--source-sha256', source_hash, '--model-id', item['finalization']['model']['model_id'], '--source-relative-path', relative, '--input-root', str(source_dir)]
                    validation = retained_envelope(validation_command, case, case, 'artifact-validator', case_deadline, token, case_deadline - time.monotonic())
                    item['artifact_validator_envelope'] = validation
                    require(validation.get('success') is True and validation.get('cleanup_verified') is True and validation.get('streams_drained') is True, 'Native artifact validator failed/unverified')
                    evidence = read_json(supplemental / 'receipt.json')
                    require(evidence.get('accepted') is True and evidence['native_result_sha256'] == item['native_result_sha256'] and evidence['script_sha256'] == item['script_sha256'], 'Native/artifact evidence binding differs')
                    require(evidence['selected_model_id'] == expected_model_id(relative)
                            and evidence['selected_source_sha256'] == source_hash
                            and evidence['selected_source_relative_path'] == relative,
                            'Supplemental selected-source binding differs')
                    after_finalization = finalized_pipeline(output, envelope, canonical, source_hash, configured, source, source_dir, relative, args.backend)
                    require(after_finalization == item['finalization'], 'Finalized pipeline bytes/identity changed after supplemental analysis')
                    require_bound_file(output, scripts[0], after_finalization['verified_pipeline_artifacts'])
                    require_bound_file(output, native[0], after_finalization['verified_pipeline_artifacts'])
                    require(digest(config) == item['config_sha256'], 'Supplemental analysis changed caller configuration')
                    item['finalized_artifacts_reverified_after_worker'] = True
                    item['native_artifact_evidence'] = evidence
                    item['artifact_inventory'] = artifact_inventory(output)
                    item['supplemental_artifact_inventory'] = artifact_inventory(supplemental)
                    require(digest(native[0]) == item['native_result_sha256'], 'Supplemental analysis changed native bytes')
                    require(bookend(case_deadline, token) == before, 'ROOT tracked bytes/HEAD changed during native case')
                    common.verify_wheel(BASE, args.split, WHEEL, ARTIFACT)
                    require(not token.cancelled and time.monotonic() < case_deadline, 'Case deadline/resource exhausted during verification')
                    item['accepted'] = True
                except Exception as error:
                    item['error'] = {'type': type(error).__name__, 'message': str(error)}
                    if (case / 'output').is_dir():
                        try:
                            item['partial_artifact_inventory'] = artifact_inventory(case / 'output')
                        except Exception as inventory_error:
                            item['partial_inventory_error'] = str(inventory_error)
                item['elapsed_seconds'] = time.monotonic() - case_started
                item['storage'] = token.snapshot()
                (case / 'receipt.json').write_text(json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + '\n')
                (root / 'receipt.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
                print(json.dumps({'case': relative, 'accepted': item['accepted'], 'error': item.get('error'), 'elapsed_seconds': item['elapsed_seconds']}, sort_keys=True), flush=True)
                if token.cancelled or time.monotonic() >= deadline:
                    break
        after = bookend(deadline, token)
        (root / 'root-after.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
        require(after == before, 'Complete verification-branch ROOT tracked bookends differ')
        record['root_tracked_files_unchanged'] = len(after['files'])
        common.verify_wheel(BASE, args.split, WHEEL, ARTIFACT)
        require(digest(WORKER) == args.worker_sha256 and digest(Path(__file__)) == record['runner_sha256'] and not token.cancelled and time.monotonic() < deadline, 'Final helper/resource/deadline admission failed')
        record['accepted'] = args.admission_only or (len(record['cases']) == 4 and all(item['accepted'] for item in record['cases']))
    except Exception as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
    if token is not None:
        record['storage'] = token.snapshot()
        if token.cancelled:
            record['accepted'] = False
    record['elapsed_seconds'] = time.monotonic() - started
    record['remaining_selected_cases'] = [relative for relative, _ in CASES[args.backend] if relative not in {item['source_relative_path'].removeprefix('input/gnn_files/') for item in record['cases']}]
    record['free_storage_bytes_at_end'] = shutil.disk_usage(BASE).free
    if record['free_storage_bytes_at_end'] <= 4 * 1024**3:
        record['accepted'] = False
        record.setdefault('error', {'type': 'ResourceCancellation', 'message': 'Postflight observed storage below authoritative four GiB floor'})
    (root / 'receipt.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'receipt': str(root / 'receipt.json'), 'accepted': record['accepted'], 'error': record.get('error')}, sort_keys=True), flush=True)
    return 0 if record['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
