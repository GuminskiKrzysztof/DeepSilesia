document.querySelectorAll('.accordion-toggle').forEach(button => {
    button.addEventListener('click', function() {
        const accordionContent = this.nextElementSibling;
        this.classList.toggle('active');
        accordionContent.style.maxHeight = accordionContent.style.maxHeight ? null : accordionContent.scrollHeight + 'px';
    });
});