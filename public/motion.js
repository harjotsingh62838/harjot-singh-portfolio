// Pointer accents never block navigation and respect reduced motion.
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
if(window.matchMedia('(hover: hover) and (pointer: fine)').matches){document.querySelectorAll('.optiongrid a,.projectcard').forEach(card=>{card.addEventListener('pointermove',event=>{if(reducedMotion.matches)return;const rect=card.getBoundingClientRect();card.style.setProperty('--mx',`${event.clientX-rect.left}px`);card.style.setProperty('--my',`${event.clientY-rect.top}px`);});});}
