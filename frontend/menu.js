// Stacked drill-down interaction adapted to the dashboard's dependency-free UI.
const trigger = document.getElementById("settings");
const popup = document.getElementById("presenter-menu");
const source = document.getElementById("source");
const scenario = document.getElementById("scenario");
const pause = document.getElementById("pause");
let frames = [],
  feedbackTimer,
  closeTimer;
const arrow = (direction) => {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(svg.namespaceURI, "path");
  path.setAttribute(
    "d",
    direction === "left" ? "m15 18-6-6 6-6" : "m9 18 6-6-6-6",
  );
  svg.append(path);
  return svg;
};
function feedback(text) {
  const el = document.getElementById("action-feedback");
  clearTimeout(feedbackTimer);
  el.textContent = text;
  el.hidden = false;
  feedbackTimer = setTimeout(() => {
    el.hidden = true;
  }, 2400);
}
function change(select, value) {
  select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}
function choices(select) {
  return Array.from(select.options, (option) => ({
    label: option.textContent,
    selected: select.value === option.value,
    action: () => {
      change(select, option.value);
      feedback(option.textContent + " selected");
    },
  }));
}
function rootItems() {
  return [
    {
      label: "Data source",
      hint: source.selectedOptions[0].textContent,
      children: () => choices(source),
    },
    {
      label: "Simulation scenario",
      disabled: source.value !== "mock",
      children: () => choices(scenario),
    },
  ];
}
function place() {
  if (popup.hidden) return;
  const rect = trigger.getBoundingClientRect();
  const width = Math.min(310, innerWidth - 24);
  popup.style.left =
    Math.max(12, Math.min(rect.right - width, innerWidth - width - 12)) + "px";
  popup.style.top = Math.min(rect.bottom + 8, innerHeight - 120) + "px";
  frames.forEach((frame) => {
    const available =
      innerHeight - popup.getBoundingClientRect().top - frame.top - 68;
    frame.panel.style.setProperty(
      "--menu-list-height",
      Math.max(44, available) + "px",
    );
  });
}
function activeButtons() {
  const panel = frames.at(-1)?.panel;
  return panel
    ? Array.from(panel.querySelectorAll(".menu-row:not(:disabled)"))
    : [];
}
function focusRow(index = 0) {
  const buttons = activeButtons();
  if (!buttons.length) return;
  const selected = (index + buttons.length) % buttons.length;
  buttons.forEach((button, i) => {
    button.tabIndex = i === selected ? 0 : -1;
  });
  buttons[selected].focus();
}
function close(returnFocus = true) {
  if (popup.hidden || trigger.getAttribute("aria-expanded") === "false") return;
  trigger.setAttribute("aria-expanded", "false");
  popup.inert = true;
  popup.setAttribute("aria-hidden", "true");
  frames.forEach((frame) => frame.panel.classList.add("leaving"));
  closeTimer = setTimeout(() => {
    popup.hidden = true;
    popup.replaceChildren();
    frames = [];
  }, 150);
  if (returnFocus) trigger.focus();
}
function popTo(depth) {
  if (depth < 0 || depth >= frames.length - 1) return;
  const restore = frames[depth + 1].fromIndex;
  const leaving = frames.splice(depth + 1);
  leaving.forEach((frame) => {
    frame.panel.inert = true;
    frame.panel.setAttribute("aria-hidden", "true");
    frame.panel.classList.add("leaving");
    setTimeout(() => frame.panel.remove(), 150);
  });
  const panel = frames.at(-1).panel;
  panel.dataset.behind = "false";
  panel
    .querySelectorAll('[aria-haspopup="menu"]')
    .forEach((row) => row.setAttribute("aria-expanded", "false"));
  panel.removeAttribute("aria-hidden");
  panel.querySelector(".menu-list").setAttribute("role", "menu");
  focusRow(restore);
  place();
}
function push(title, items, top = 0, fromIndex = 0) {
  const depth = frames.length;
  frames.forEach((frame) => {
    frame.panel.dataset.behind = "true";
    frame.panel.setAttribute("aria-hidden", "true");
    frame.panel.querySelector(".menu-list").removeAttribute("role");
    frame.panel.querySelectorAll("button").forEach((button) => {
      button.tabIndex = -1;
    });
  });
  const panel = document.createElement("div");
  panel.className = "menu-panel" + (depth ? " nested" : "");
  panel.style.top = top + "px";
  panel.style.zIndex = 10 + depth;
  const heading = document.createElement("div");
  heading.className = "menu-heading";
  if (depth) {
    const back = document.createElement("button");
    back.className = "menu-back";
    back.setAttribute("aria-label", "Back to presenter controls");
    back.append(arrow("left"));
    back.onclick = (event) => {
      event.stopPropagation();
      popTo(depth - 1);
    };
    heading.append(back);
  }
  heading.append(document.createTextNode(title));
  const list = document.createElement("div");
  list.className = "menu-list";
  list.setAttribute("role", "menu");
  list.setAttribute("aria-label", title);
  items.forEach((item) => {
    const row = document.createElement("button");
    row.className = "menu-row";
    row.type = "button";
    row.tabIndex = -1;
    row.disabled = !!item.disabled;
    row.setAttribute("role", "selected" in item ? "menuitemradio" : "menuitem");
    if ("selected" in item)
      row.setAttribute("aria-checked", String(item.selected));
    if (item.children) {
      row.setAttribute("aria-haspopup", "menu");
      row.setAttribute("aria-expanded", "false");
    }
    const label = document.createElement("span");
    label.className = "menu-label";
    label.textContent = item.label;
    row.append(label);
    if (item.children) row.append(arrow("right"));
    else if (item.selected) {
      const check = document.createElement("img");
      check.src = "assets/check.svg";
      check.alt = "Selected";
      row.append(check);
    }
    row.onclick = (event) => {
      event.stopPropagation();
      if (frames.at(-1)?.panel !== panel) {
        popTo(depth);
        return;
      }
      if (item.children) {
        row.setAttribute("aria-expanded", "true");
        const index = activeButtons().indexOf(row);
        const anchor =
          row.getBoundingClientRect().top - popup.getBoundingClientRect().top;
        push(item.label, item.children(), anchor, index);
      } else {
        item.action();
        close();
      }
    };
    row.onfocus = () => {
      if (frames.at(-1)?.panel === panel)
        activeButtons().forEach((button) => {
          button.tabIndex = button === row ? 0 : -1;
        });
    };
    list.append(row);
  });
  panel.append(heading, list);
  panel.onclick = () => {
    if (frames.at(-1)?.panel !== panel) popTo(depth);
  };
  popup.append(panel);
  frames.push({ panel, top, fromIndex });
  place();
  focusRow();
}
function open() {
  clearTimeout(closeTimer);
  popup.replaceChildren();
  frames = [];
  popup.hidden = false;
  popup.inert = false;
  popup.removeAttribute("aria-hidden");
  trigger.setAttribute("aria-expanded", "true");
  push("Presenter controls", rootItems());
}
trigger.onclick = () =>
  trigger.getAttribute("aria-expanded") === "true" ? close() : open();
trigger.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    open();
    if (event.key === "ArrowUp") focusRow(-1);
  }
});
popup.addEventListener("keydown", (event) => {
  const buttons = activeButtons();
  const index = buttons.indexOf(document.activeElement);
  switch (event.key) {
    case "ArrowDown":
      event.preventDefault();
      focusRow(index + 1);
      break;
    case "ArrowUp":
      event.preventDefault();
      focusRow(index - 1);
      break;
    case "Home":
      event.preventDefault();
      focusRow(0);
      break;
    case "End":
      event.preventDefault();
      focusRow(-1);
      break;
    case "ArrowRight":
      if (document.activeElement?.hasAttribute("aria-haspopup")) {
        event.preventDefault();
        document.activeElement.click();
      }
      break;
    case "ArrowLeft":
    case "Backspace":
      if (frames.length > 1) {
        event.preventDefault();
        popTo(frames.length - 2);
      }
      break;
    case "Escape":
      event.preventDefault();
      frames.length > 1 ? popTo(frames.length - 2) : close();
      break;
    case "Tab":
      close(false);
      break;
  }
});
document.addEventListener("pointerdown", (event) => {
  if (
    !popup.hidden &&
    !popup.contains(event.target) &&
    !trigger.contains(event.target)
  )
    close(false);
});
window.addEventListener("resize", place);
window.addEventListener(
  "scroll",
  () => {
    if (!popup.hidden) close(false);
  },
  { passive: true },
);
pause.addEventListener("click", () =>
  feedback(
    pause.textContent.includes("Resume")
      ? "Simulation paused"
      : "Simulation resumed",
  ),
);
const fullscreen = document.getElementById("fullscreen");
document.addEventListener("fullscreenchange", () => {
  const active =
    document.fullscreenElement === document.getElementById("inspection-video");
  fullscreen.setAttribute(
    "aria-label",
    active ? "Exit fullscreen" : "Enter fullscreen",
  );
  fullscreen.title = active ? "Exit fullscreen" : "Enter fullscreen";
  fullscreen.setAttribute("aria-pressed", String(active));
  feedback(
    active ? "Fullscreen enabled · press Esc to exit" : "Fullscreen exited",
  );
});
document.addEventListener("fullscreenerror", () =>
  feedback("Fullscreen unavailable. Use your browser’s fullscreen command."),
);
