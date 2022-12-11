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
        waitQuerySelector('#additioanl-tags'),
        waitQuerySelector('#exclude-tags'),

        // tag-confident labels
        waitQuerySelector('#rating-confidents'),
        waitQuerySelector('#tag-confidents')
    ]).then(elements => {

        const $additionalTags = elements[0].querySelector('textarea')
        const $excludeTags = elements[1].querySelector('textarea')
        const $ratingConfidents = elements[2]
        const $tagConfidents = elements[3]

        let $selectedTextarea = $additionalTags

        /**
         * @this {HTMLElement}
         * @param {MouseEvent} e
         * @listens document#click
         */
        function onClickTextarea(e) {
            $selectedTextarea = this
        }

        $additionalTags.addEventListener('click', onClickTextarea)
        $excludeTags.addEventListener('click', onClickTextarea)

        /**
         * @this {HTMLElement}
         * @param {MouseEvent} e
         * @listens document#click
         */
        function onClickLabels(e) {
            // find clicked label item's wrapper element
            const $tag = e.target.closest('.output-label > div:not(:first-child)')
            if (!$tag) {
                return
            }

            /** @type {string} */
            const tag = $tag.querySelector('.leading-snug').textContent

            // ignore if tag is already exist in textbox
            const escapedTag = tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
            const pattern = new RegExp(`(^|,)\\s{0,}${escapedTag}\\s{0,}($|,)`)
            if (pattern.test($selectedTextarea.value)) {
                return
            }

            if ($selectedTextarea.value !== '') {
                $selectedTextarea.value += ', '
            }

            $selectedTextarea.value += tag
        }

        $ratingConfidents.addEventListener('click', onClickLabels)
        $tagConfidents.addEventListener('click', onClickLabels)

    }).catch(err => {
        console.error(err)
    })
})