// Site-local override of Hextra v0.2.9 assets/js/theme.js.
// Single behavioral change vs upstream: the no-preference default mirrors the
// boot script in layouts/partials/custom/head-end.html — DARK unless the OS
// explicitly prefers LIGHT (upstream required an explicit OS dark preference).
// Everything else (button sync, click handling, storage contract) is upstream.
// On a Hextra version bump, re-diff against the module's theme.js.

const themeToggleButtons = document.querySelectorAll(".theme-toggle");

// Change the icons inside the button based on previous settings
if (
  localStorage.getItem("color-theme") === "dark" ||
  (!("color-theme" in localStorage) &&
    !window.matchMedia("(prefers-color-scheme: light)").matches)
) {
  themeToggleButtons.forEach((el) => el.dataset.theme = "dark");
} else {
  themeToggleButtons.forEach((el) => el.dataset.theme = "light");
}

themeToggleButtons.forEach((el) => {
  el.addEventListener("click", function () {
    if (localStorage.getItem("color-theme")) {
      if (localStorage.getItem("color-theme") === "light") {
        document.documentElement.classList.add("dark");
        document.documentElement.style.colorScheme = "dark";
        localStorage.setItem("color-theme", "dark");
      } else {
        document.documentElement.classList.remove("dark");
        document.documentElement.style.colorScheme = "light";
        localStorage.setItem("color-theme", "light");
      }
    } else {
      if (document.documentElement.classList.contains("dark")) {
        document.documentElement.classList.remove("dark");
        document.documentElement.style.colorScheme = "light";
        localStorage.setItem("color-theme", "light");
      } else {
        document.documentElement.classList.add("dark");
        document.documentElement.style.colorScheme = "dark";
        localStorage.setItem("color-theme", "dark");
      }
    }
    el.dataset.theme = document.documentElement.classList.contains("dark") ? "dark" : "light";
  });
});
