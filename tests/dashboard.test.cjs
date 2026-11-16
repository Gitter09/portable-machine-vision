// Offline state/contract tests. No browser or third-party packages required.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function app() {
  const elements = new Map();
  const context2d = new Proxy({}, { get: (_, key) => key === 'measureText' ? () => ({ width: 100 }) : () => {} });
  const element = () => ({ hidden: false, textContent: '', children: [], width:1280, height:720,
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute() {}, removeAttribute() {}, getContext() { return { ...context2d, canvas: {width:1280,height:720}, ...new Proxy({}, {}) }; } });
  const get = id => {
    if (!elements.has(id)) {
      const e = element();
      e.getContext = () => new Proxy({ canvas: e }, { get: (target,key) => key in target ? target[key] : context2d[key], set: (target,key,value) => {target[key]=value;return true;} });
      elements.set(id,e);
    }
    return elements.get(id);
  };
  const sandbox = { document: {getElementById:get, createElement:element}, setTimeout:()=>{}, Date, console, AbortSignal };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync('frontend/app.js','utf8'),sandbox);
  return { run:code=>vm.runInContext(code,sandbox), get };
}
test('pause button switches its icon and label on each click', () => {
  const {run,get} = app();
  run("document.getElementById('pause').onclick()");
  assert.equal(get('pause').children[0].src, 'assets/play.svg');
  assert.equal(get('pause').children[1], 'Resume simulation');
  run("document.getElementById('pause').onclick()");
  assert.equal(get('pause').children[0].src, 'assets/pause.svg');
  assert.equal(get('pause').children[1], 'Pause simulation');
});
test('pause button is hidden for Live and restored for simulation', () => {
  const {run,get} = app();
  run("document.getElementById('source').onchange({target:{value:'live'}})");
  assert.equal(get('pause').hidden, true);
  run("document.getElementById('source').onchange({target:{value:'mock'}})");
  assert.equal(get('pause').hidden, false);
});
test('fullscreen requests only the inspection viewport', async () => {
  const {run} = app();
  run("document.getElementById('inspection-video').requestFullscreen = async () => { document.requestedTarget = 'inspection-video'; }");
  await run("document.getElementById('fullscreen').onclick()");
  assert.equal(run('document.requestedTarget'), 'inspection-video');
});
test('simulation preserves four-class IDs for every component state', () => {
  const {run} = app();
  for (const [scenario, expected] of [['present',[2,1]],['ic',[0,1]],['terminal',[2,3]],['both',[0,3]],['unknown',[]]]) {
    const result = run(`scenario='${scenario}';mockResult()`);
    assert.deepEqual(Array.from(result.detections,d=>d.class_id),expected);
  }
});
test('stale, unmatched, and disconnected hide overlays and current findings', () => {
  const {run,get} = app();
  for (const state of ['stale','unmatched','disconnected']) {
    run(`renderStatus({state:'${state}',detections:[{class_id:2,score:.99}]})`);
    assert.equal(get('empty').hidden,false);
    assert.equal(get('count').textContent,'No current findings');
    assert.equal(get('detections').children[0].children[1].children[0].textContent,'Unknown');
  }
});
test('conflicting predictions do not silently choose a present state', () => {
  const {run,get} = app();
  run("renderStatus({state:'matched',detections:[{class_id:2,score:.99},{class_id:0,score:.9}]})");
  assert.equal(get('detections').children[0].children[1].children[0].textContent,'Conflicting');
});
