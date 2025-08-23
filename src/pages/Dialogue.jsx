const response = await fetch("https://api.render.com/deploy/srv-d2kl0bruibrs73eaac30?key=RhnbCDaBN4c", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message }),
});
