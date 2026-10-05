"""Deferred lightweight adversarial controls; require fresh root resource GO.

No tests/imports/probes run at import. This script must itself be launched under
one public process envelope/deadline; it never edits product or model sources.
"""
from pathlib import Path
import argparse,copy,hashlib,html,importlib.util,json,marshal,math,os,shutil,struct,time,types,uuid
CODE=Path(__file__).resolve().parent
BASE=Path(os.environ["GNN_NATIVE_VERIFY_WORKSPACE"]).absolute()
RUNNER_SHA="5c8e09d68e0e37db2249ce3fb09b1d2419155013dca661ec2d052c7f951dce0c"
WORKER_SHA="4b75739ce163eecf5cbbdd9439db0848bf811d20c84829264ad2f26bc985b45c"

def load_exact(name,digest):
 path=CODE/name;raw=path.read_bytes()
 assert not path.is_symlink() and hashlib.sha256(raw).hexdigest()==digest
 module=types.ModuleType(name);module.__file__=str(path)
 exec(compile(raw,str(path),"exec",dont_inherit=True),module.__dict__)
 return module

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--resource-go",required=True);args=parser.parse_args()
 assert args.resource_go=="root-reviewed-lightweight-controls"
 assert shutil.disk_usage(BASE).free>=6*1024**3,"Storage HOLD before any installed import/control"
 started=time.monotonic();deadline=started+60
 out=BASE/"evidence"/("reported-native-controls-"+uuid.uuid4().hex[:12]);out.mkdir();rows=[]
 runner=load_exact("runner.py",RUNNER_SHA);worker=load_exact("artifacts.py",WORKER_SHA)
 def check(name,fn,refuses=False):
  assert time.monotonic()<deadline and shutil.disk_usage(BASE).free>4*1024**3,"Deadline/resource exhausted"
  error=None
  try:fn()
  except Exception as e:error=type(e).__name__+":"+str(e)
  rows.append({"name":name,"expected_refusal":refuses,"passed":(error is not None)==refuses,"error":error})
 fixture=out/"loader_fixture.py";fixture.write_text("SENTINEL='checked-source-bytes'\n")
 original_common,original_common_sha=runner.COMMON,runner.COMMON_SHA
 runner.COMMON=fixture;runner.COMMON_SHA=hashlib.sha256(fixture.read_bytes()).hexdigest()
 # A substituted import cache must never affect the digest-checked source loader.
 cache=Path(importlib.util.cache_from_source(str(fixture)));cache.parent.mkdir()
 observed=fixture.stat();header=importlib.util.MAGIC_NUMBER+struct.pack("<III",0,int(observed.st_mtime),observed.st_size)
 cache.write_bytes(header+marshal.dumps(compile("SENTINEL='substituted-cache'\n",str(fixture),"exec")))
 def cache_control():
  spec=importlib.util.spec_from_file_location("known_vulnerable_loader_control",fixture)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  worker.require(module.SENTINEL=="substituted-cache","Adversarial cache control did not reproduce loader substitution")
 check("valid timestamp-bound substituted pyc reaches conventional loader",cache_control)
 check("digest-checked source bypasses valid substituted pyc",lambda:worker.require(runner.load_common().SENTINEL=="checked-source-bytes","Wrong source executed"))
 fixture.write_text("SENTINEL='changed-unreviewed'\n")
 check("changed common source refused",runner.load_common,True)
 fixture.write_text("SENTINEL='checked-source-bytes'\n")
 symlink=out/"linked_helper.py";symlink.symlink_to(fixture);runner.COMMON=symlink
 check("common symlink refused",runner.load_common,True)
 runner.COMMON,runner.COMMON_SHA=original_common,original_common_sha
 native={"schema_version":"numpyro_simulation_v1","model_name":"numpyro_pomdp","num_states":2,"num_observations":3,"num_actions":4,"num_timesteps":2,"validation":{"all_valid":True},"beliefs":[[.6,.4],[.4,.6]],"observations":[0,2],"actions":[0,3]}
 emitted=out/"numpyro_emitted.py";emitted.write_text("def run_simulation():\n    num_states=2\n    num_obs=3\n    num_actions=4\n    results={'model_name':'numpyro_pomdp'}\n")
 def validate(data=native,path=emitted,backend="numpyro"):
  views,_=worker.strict_views(data,2,backend);return worker.emitted_binding(data,path,views,backend)
 check("generic display label with external selection is valid",validate)
 for name,key,value in [("observation upper bound from emitted m=3","observations",[0,3]),("action upper bound from emitted p=4","actions",[0,4]),("negative categorical index","actions",[-1,0]),("fractional categorical index","observations",[.5,0]),("Boolean categorical index","actions",[False,0]),("input length mismatch","observations",[0]),("different native display label","model_name","other"),("native cardinality differs","num_observations",4),("posterior width differs","beliefs",[[.2,.3,.5],[.2,.3,.5]]),("negative probabilities","beliefs",[[-.1,1.1],[.4,.6]]),("zero mass","beliefs",[[0,0],[.4,.6]]),("nonunit mass","beliefs",[[.6,.6],[.4,.6]]),("nonfinite posterior","beliefs",[[float('nan'),.4],[.4,.6]]),("ragged posterior","beliefs",[[.6,.4],[1]]),("Boolean T","num_timesteps",True),("false native validation","validation",{"all_valid":False}),("continuous declaration conflicts with reported categorical schema","model_kind","continuous"),("Gaussian covariance conflicts with reported categorical schema","posterior_cov",[[[1,0],[0,1]],[[1,0],[0,1]]]),("malformed semantic metadata","runtime_metadata","bad")]:
  data=copy.deepcopy(native);data[key]=value;check(name,lambda data=data:validate(data),True)
 flat=out/"flat.jl";flat.write_text('const MODEL_NAME = "rx_model"\nconst SCHEMA_VERSION = "rxinfer_simulation_v1"\nconst NUM_STATES = 2\nconst NUM_OBSERVATIONS = 3\nconst NUM_ACTIONS = 4\n')
 rx=copy.deepcopy(native);rx.update(schema_version="rxinfer_simulation_v1",model_name="rx_model",runtime_metadata={"julia_version":"1.12.7","rxinfer_version":"5.5.0"},model_parameters={"inference_iterations":20})
 check("flat RxInfer emitted bounds positive",lambda:validate(rx,flat,"rxinfer"))
 for name,key,value in [("RxInfer wrong name","model_name","wrong"),("RxInfer observation over upper","observations",[0,3]),("RxInfer action over upper","actions",[0,4]),("RxInfer incomplete iterations","model_parameters",{"inference_iterations":19}),("RxInfer wrong dependency","runtime_metadata",{"julia_version":"1.12.7","rxinfer_version":"5.4.0"})]:
  data=copy.deepcopy(rx);data[key]=value;check(name,lambda data=data:validate(data,flat,"rxinfer"),True)
 multi=out/"multi.jl";agents=["agent_a","agent_b"]
 likelihoods=[[[1/3,1/3] for _ in range(3)],[[.5,.5,.5],[.5,.5,.5]]]
 transitions=[[[[float(i==j) for _ in range(actions)] for j in range(states)] for i in range(states)] for states,actions in ((2,2),(3,4))]
 priors=[[.5,.5],[1/3,1/3,1/3]]
 constants={"MODEL_NAME":"rx_agents","SCHEMA_VERSION":"rxinfer_stigmergic_swarm_v1","NUM_AGENTS":2,"AGENTS":agents,"AGENT_AS":likelihoods,"AGENT_BS":transitions,"AGENT_DS":priors}
 multi.write_text("\n".join("const "+name+" = "+json.dumps(value) for name,value in constants.items())+"\n")
 coupled=copy.deepcopy(rx)
 for key in ("beliefs","actions","observations","num_states","num_observations","num_actions"):coupled.pop(key,None)
 coupled.update(schema_version="rxinfer_stigmergic_swarm_v1",model_name="rx_agents",agents=agents,beliefs_by_agent={"agent_a":[[.5,.5],[.6,.4]],"agent_b":[[1/3,1/3,1/3],[.2,.3,.5]]},observations_by_agent={"agent_a":[0,2],"agent_b":[0,1]},actions_by_agent={"agent_a":[0,1],"agent_b":[0,3]})
 check("heterogeneous named agent emitted bounds positive",lambda:validate(coupled,multi,"rxinfer"))
 for name,key,value in [("agent observation distinct m bound","observations_by_agent",{"agent_a":[0,3],"agent_b":[0,1]}),("agent action distinct p bound","actions_by_agent",{"agent_a":[0,2],"agent_b":[0,3]}),("renamed native agent identity","agents",["agent_a","renamed_b"]),("wrong per-agent posterior cardinality","beliefs_by_agent",{"agent_a":[[.5,.5],[.6,.4]],"agent_b":[[.5,.5],[.5,.5]]})]:
  data=copy.deepcopy(coupled);data[key]=value;check(name,lambda data=data:validate(data,multi,"rxinfer"),True)
 for label,value in (("Boolean named-agent posterior",[[True,False],[False,True]]),("numeric-string named-agent posterior",[["0.5","0.5"],["0.6","0.4"]])):
  data=copy.deepcopy(coupled);data["beliefs_by_agent"]["agent_a"]=value
  check(label,lambda data=data:validate(data,multi,"rxinfer"),True)
 # Actual parent seam controls construct finalized receipts around copied files.
 # Mutations update matching context/detail IDs together to expose false agreement.
 canonical={"3_gnn.py":3,"7_export.py":7,"8_visualization.py":8,"11_render.py":11,"12_execute.py":12,"16_analysis.py":16,"20_website.py":20}
 def parent_fixture():
  root=out/("parent-"+uuid.uuid4().hex[:8]);output=root/"output";summarydir=output/"00_pipeline_summary";summarydir.mkdir(parents=True)
  input_root=root/"input/gnn_files";source=input_root/"nested/model.md";source.parent.mkdir(parents=True);source.write_text("Authored model bytes\n")
  relative="nested/model.md";source_sha=runner.digest(source);model_id=runner.expected_model_id(relative);config={"pipeline":{"parallel":{"enabled":False}}}
  model={"source_path":str(source),"relative_path":relative,"model_id":model_id,"sha256":source_sha,"artifact_stem":"model"}
  context={"run_id":"parent-fixture","input_root":str(input_root),"output_root":str(output),"frameworks":["numpyro"],"config_json":json.dumps(config,sort_keys=True),"models":[model]}
  detail={"source_path":str(source),"source_relative_path":relative,"model_id":model_id,"source_sha256":source_sha,"framework":"numpyro","success":True,"skipped":False,"cleanup_verified":True,"streams_drained":True,"cancelled":False}
  execution={"run_id":"parent-fixture","success":True,"status":"success","successful_executions":1,"failed_executions":0,"skipped_executions":0,"attempted_scripts":1,"total_scripts":1,"execution_details":[detail]}
  steps=[];units=[]
  for name in canonical:
   artifact=output/(name+".evidence");artifact.write_text("Final step evidence "+name)
   item={"path":artifact.relative_to(output).as_posix(),"sha256":runner.digest(artifact)}
   steps.append({"script_name":name,"status":"SUCCESS","run_id":"parent-fixture","artifacts":[item]})
   units.append({"unit_id":name,"status":"DONE","input_identity":{"run_id":"parent-fixture"},"artifact_hashes":{item["path"]:item["sha256"]}})
  integrity={"status":"verified"};session={"session_id":"parent-fixture","final_status":"SUCCESS","evidence_integrity":integrity,"units":units}
  session_path=summarydir/"run_session.json";session_path.write_text(json.dumps(session))
  summary={"run_id":"parent-fixture","end_time":"fixture-finalized","overall_status":"SUCCESS","unfinished_steps":[],"evidence_integrity":integrity,"run_session_sha256":runner.digest(session_path),"steps":steps,"planned_steps":list(canonical)}
  (summarydir/"pipeline_execution_summary.json").write_text(json.dumps(summary));(summarydir/"run_context.json").write_text(json.dumps(context))
  execpath=output/"12_execute_output/summaries/execution_summary.json";execpath.parent.mkdir(parents=True);execpath.write_text(json.dumps(execution))
  envelope={"return_code":0,"cancelled":False,"error_type":None,"cleanup_verified":True,"streams_drained":True}
  return output,envelope,source_sha,config,source,input_root,relative,context,execution
 def parent_control(mutate=None):
  output,envelope,source_sha,config,source,input_root,relative,context,execution=parent_fixture()
  if mutate:mutate(output,source,context,execution)
  (output/"00_pipeline_summary/run_context.json").write_text(json.dumps(context));(output/"12_execute_output/summaries/execution_summary.json").write_text(json.dumps(execution))
  return runner.finalized_pipeline(output,envelope,canonical,source_sha,config,source,input_root,relative,"numpyro")
 check("parent finalized path-derived selection positive",parent_control)
 def forged_ids(output,source,context,execution):context["models"][0]["model_id"]=execution["execution_details"][0]["model_id"]="matching-forged-ID"
 check("matching forged context and execution model IDs refused",lambda:parent_control(forged_ids),True)
 def forged_relative(output,source,context,execution):context["models"][0]["relative_path"]=execution["execution_details"][0]["source_relative_path"]="other/model.md"
 check("matching forged source relative paths refused",lambda:parent_control(forged_relative),True)
 def forged_source(output,source,context,execution):context["models"][0]["source_path"]=execution["execution_details"][0]["source_path"]=str(source.parent/"other.md")
 check("matching forged physical source paths refused",lambda:parent_control(forged_source),True)
 check("changed selected source bytes refused",lambda:parent_control(lambda output,source,context,execution:source.write_text("Changed source\n")),True)
 check("wrong resolved caller configuration refused",lambda:parent_control(lambda output,source,context,execution:context.update(config_json=json.dumps({"different":True}))),True)
 check("wrong configured backend refused",lambda:parent_control(lambda output,source,context,execution:context.update(frameworks=["rxinfer"])),True)
 check("wrong execution backend refused",lambda:parent_control(lambda output,source,context,execution:execution["execution_details"][0].update(framework="rxinfer")),True)
 check("wrong configured output root refused",lambda:parent_control(lambda output,source,context,execution:context.update(output_root=str(output.parent/"unrelated"))),True)
 def bound_control(kind):
  output,envelope,source_sha,config,source,input_root,relative,context,execution=parent_fixture()
  admitted=runner.finalized_pipeline(output,envelope,canonical,source_sha,config,source,input_root,relative,"numpyro")
  candidate=output/"3_gnn.py.evidence"
  if kind=="unmanifested":candidate=output/"unmanifested-native.json";candidate.write_text("{}")
  elif kind=="mutated":candidate.write_text("Same finalized pathname, changed bytes")
  elif kind=="symlink":candidate.unlink();candidate.symlink_to(source)
  return runner.require_bound_file(output,candidate,admitted["verified_pipeline_artifacts"])
 check("actual parent manifest member positive",lambda:bound_control("good"))
 check("unmanifested native candidate refused",lambda:bound_control("unmanifested"),True)
 check("manifested same-path byte mutation refused",lambda:bound_control("mutated"),True)
 check("manifested symlink replacement refused",lambda:bound_control("symlink"),True)
 def changed_after_worker():
  output,envelope,source_sha,config,source,input_root,relative,context,execution=parent_fixture()
  first=runner.finalized_pipeline(output,envelope,canonical,source_sha,config,source,input_root,relative,"numpyro")
  (output/"16_analysis.py.evidence").write_text("Changed after supplemental analysis")
  return runner.finalized_pipeline(output,envelope,canonical,source_sha,config,source,input_root,relative,"numpyro")
 check("second actual finalization refuses artifact mutation after worker",changed_after_worker,True)
 views,_=worker.strict_views(native,2,"numpyro");view=next(iter(views));payload={"n_steps":2,**{key:views[view][key] for key in ("beliefs","actions","observations")}}
 good=out/"good.html";good.write_text('<html><script>const DATA = '+json.dumps(payload)+';</script></html>')
 check("HTML bound full trace positive",lambda:worker.html_witness(good,views,2))
 wrong=out/"wrong.html";value=copy.deepcopy(payload);value["beliefs"][0]=[.5,.5];wrong.write_text('<html><script>const DATA = '+json.dumps(value)+';</script></html>')
 check("HTML same-shape value corruption refused",lambda:worker.html_witness(wrong,views,2),True)
 wrong.write_text('<html><script>const DATA = '+json.dumps(payload))
 check("HTML unfinished script refused",lambda:worker.html_witness(wrong,views,2),True)
 receipt={"accepted":all(row["passed"] for row in rows),"controls":rows,"elapsed_seconds":time.monotonic()-started,"runner_sha256":RUNNER_SHA,"worker_sha256":WORKER_SHA,"native_inference_executed":False,"scope":"Lightweight custody/index/display/trace controls; no eight-model native acceptance or Gaussian claim."}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n");print(json.dumps({"accepted":receipt["accepted"],"controls":len(rows),"directory":str(out)}));return 0 if receipt["accepted"] else 1

if __name__=="__main__":raise SystemExit(main())
