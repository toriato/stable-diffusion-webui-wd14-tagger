/**
 * wait until element is loaded and returns
 * @param {string} selector
 * @param {number} timeout 
 * @param {Element} $rootElement
 * @returns {Promise<HTMLElement>}
 */
function waitQuerySelector(selector, timeout = 5000, $rootElement = gradioApp()) {
    return new Promise((resolve, reject) => {
        const element = $rootElement.querySelector(selector)
        if (document.querySelector(element)) {
            return resolve(element)
        }

        let timeoutId

        const observer = new MutationObserver(() => {
            const element = $rootElement.querySelector(selector)
            if (!element) {
                return
            }

            if (timeoutId) {
                clearInterval(timeoutId)
            }

            observer.disconnect()
            resolve(element)
        })

        timeoutId = setTimeout(() => {
            observer.disconnect()
            reject(new Error(`timeout, cannot find element by '${selector}'`))
        }, timeout)

        observer.observe($rootElement, {
            childList: true,
            subtree: true
        })
    })
}

document.addEventListener('DOMContentLoaded', () => {
    Promise.all([
        // option texts
        waitQuerySelector('#additional-tags'),
        waitQuerySelector('#exclude-tags'),
        waitQuerySelector('#search-tags'),
        waitQuerySelector('#replace-tags'),

        // tag-confident labels
        waitQuerySelector('#rating-confidents'),
        waitQuerySelector('#tag-confidents')
    ]).then(elements => {

        const $additionalTags = elements[0];
        const $excludeTags = elements[1];
        const $searchTags = elements[2];
        const $replaceTags = elements[3];
        const $ratingConfidents = elements[4];
        const $tagConfidents = elements[5];

        let $selectedTextarea = $additionalTags;

        /**
         * @this {HTMLElement}
         * @param {MouseEvent} e
         * @listens document#click
         */
        function onClickTextarea(e) {
            $selectedTextarea = this;
        }

        $additionalTags.addEventListener('click', onClickTextarea);
        $excludeTags.addEventListener('click', onClickTextarea);
        $searchTags.addEventListener('click', onClickTextarea);
        $replaceTags.addEventListener('click', onClickTextarea);

        /**
         * @this {HTMLElement}
         * @param {MouseEvent} e
         * @listens document#click
         */
        function onClickLabels(e) {
            // find clicked label item's wrapper element
            let tag = e.target.innerText;

            // when clicking unlucky, you get all tags and percentages. Prevent inserting those here.
            const multiTag = new RegExp(`\\n.*\\n`);
            if (tag.match(multiTag)) {
                return;
            }

            // when clicking on the dotted line or the percentage, you get the percentage as well. Don't include it in the tags.
            // use this fact to choose whether to insert in positive or negative. May require some getting used to, but saves
            // having to select the input field.
            const pctPattern = new RegExp(`\\n?([0-9.]+)%$`);
            let percentage = tag.match(pctPattern);
            if (percentage) {
                tag = tag.replace(pctPattern, '');
                if (tag == '') {
                    //percentage = percentage[1];
                    // could trigger a set Thresold value event
                    return;
                }
                // when clicking on athe dotted line, insert in either the exclude or replace list
                // when not clicking on the dotted line, insert in the additingal or search list
                if ($selectedTextarea == $additionalTags) {
                    $selectedTextarea = $excludeTags;
                } else if ($selectedTextarea == $searchTags) {
                    $selectedTextarea = $replaceTags;
                }
            } else if ($selectedTextarea == $excludeTags) {
                $selectedTextarea = $additionalTags;
            } else if ($selectedTextarea == $replaceTags) {
                $selectedTextarea = $searchTags;
            }

            let value = $selectedTextarea.querySelector('textarea').value;
            if ($selectedTextarea != $replaceTags) {
                // ignore if tag is already exist in textbox
                const escapedTag = tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const pattern = new RegExp(`(^|,)\\s{0,}${escapedTag}\\s{0,}($|,)`);
                if (pattern.test(value)) {
                    return;
                }
            }

            // besides setting the value an event needs to be triggered or the value isn't actually stored.
            const spaceOrAlreadyWithComma = new RegExp(`(^|.*,)\\s*$`);
            if (!spaceOrAlreadyWithComma.test(value)) {
                value += ', ';
            }
            const input_event = new Event('input');
            $selectedTextarea.querySelector('textarea').value = value + tag;
            $selectedTextarea.querySelector('textarea').dispatchEvent(input_event);

        }

        $ratingConfidents.addEventListener('click', onClickLabels)
        $tagConfidents.addEventListener('click', onClickLabels)

    }).catch(err => {
        console.error(err)
    })
})
